import difflib
import logging
import random
import socket
import time
import urllib.error
import uuid
from dataclasses import dataclass

from pydantic import ValidationError

from resource_lock import DEFAULT_RESOURCE_LOCK_MANAGER, ResourceLease
from tool_execution_policy import PolicyAction, ToolExecutionPolicy
from tool_result import ERROR_POLICIES, ErrorCode, ToolResult
from tools import (
    TOOLS,
    ToolAuthenticationError,
    UnsafeRequestError,
    validate_tool_arguments,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedToolExecution:
    name: str
    args: dict
    tool: object
    resources: tuple


class ToolExecutor:
    # 初始化工具执行器
    # registry：工具注册表，默认使用全局 TOOLS
    # sleeper：等待函数，测试时可注入替代实现
    # jitter_fn：随机抖动函数，测试时可注入替代实现
    # lock_manager：为单独和批量工具执行提供共享资源租约的锁管理器
    # execution_policy：在资源调度前判断工具副作用是否允许的统一策略
    def __init__(
        self,
        registry=None,
        sleeper=None,
        jitter_fn=None,
        lock_manager=None,
        execution_policy=None,
    ):
        self.registry = TOOLS if registry is None else registry
        self.sleeper = sleeper or time.sleep
        self.jitter_fn = jitter_fn or random.uniform
        self.lock_manager = lock_manager or DEFAULT_RESOURCE_LOCK_MANAGER
        self.execution_policy = execution_policy or ToolExecutionPolicy()

    # 查找与未知名称最接近的少量工具
    # name：模型生成的工具名称
    # limit：最多返回的建议数量
    def _suggest_tools(self, name, limit=3):
        return difflib.get_close_matches(name, self.registry.keys(), n=limit, cutoff=0.3)

    # 查找工具、校验参数并解析本次调用需要的资源
    # name：工具注册名称
    # args：模型生成的工具参数
    def prepare(self, name, args):
        started_at = time.perf_counter()

        # 1. 从注册表查找工具，名称错误时直接返回相近工具建议
        tool = self.registry.get(name)
        if not tool:
            return ToolResult.failure(
                name,
                ErrorCode.UNKNOWN_TOOL,
                f"未知工具：{name}",
                duration_ms=self._elapsed_ms(started_at),
                suggestions=self._suggest_tools(name),
            )

        # 2. 根据 Tool Schema 校验参数，校验失败时不执行工具
        try:
            validate_tool_arguments(tool.schema, args)
        except (ValueError, SyntaxError, ArithmeticError) as error:
            return ToolResult.failure(
                name,
                ErrorCode.INVALID_ARGUMENTS,
                str(error),
                duration_ms=self._elapsed_ms(started_at),
            )

        # 3. 在资源调度前由程序策略判断本次副作用是否允许执行
        # policy_decision：当前工具副作用等级对应的执行策略决定
        policy_decision = self.execution_policy.evaluate(tool, args)
        logger.info(
            "🛡️ 工具策略: %s，副作用=%s，决定=%s",
            name,
            tool.side_effect_level.value,
            policy_decision.action.value,
        )
        if policy_decision.action == PolicyAction.REQUIRE_APPROVAL:
            return ToolResult.failure(
                name,
                ErrorCode.APPROVAL_REQUIRED,
                policy_decision.reason,
                duration_ms=self._elapsed_ms(started_at),
            )
        if policy_decision.action == PolicyAction.DENY:
            return ToolResult.failure(
                name,
                ErrorCode.POLICY_DENIED,
                policy_decision.reason,
                duration_ms=self._elapsed_ms(started_at),
            )

        # 4. 根据工具参数解析读、写或独占资源，供批量调度和执行锁保护
        try:
            # resources：工具本次调用需要一次性获取的资源声明
            resources = tool.resolve_resources(args)
        except UnsafeRequestError as error:
            return ToolResult.failure(
                name,
                ErrorCode.UNSAFE_REQUEST,
                str(error),
                duration_ms=self._elapsed_ms(started_at),
            )
        except (ValueError, SyntaxError, ArithmeticError) as error:
            return ToolResult.failure(
                name,
                ErrorCode.INVALID_ARGUMENTS,
                str(error),
                duration_ms=self._elapsed_ms(started_at),
            )

        return PreparedToolExecution(name, args, tool, resources)

    # 准备并执行单个工具调用，保留原有公开入口
    # name：工具注册名称
    # args：模型生成的工具参数
    def execute(self, name, args):
        # prepared：工具预处理结果或无需执行的失败 ToolResult
        prepared = self.prepare(name, args)
        if isinstance(prepared, ToolResult):
            return prepared
        return self.execute_prepared(prepared)

    # 在资源租约保护下执行已完成预处理的工具调用
    # prepared：已通过工具查找、参数校验和资源解析的执行对象
    # resource_lease：批量调度器预先获得的资源租约
    def execute_prepared(self, prepared, resource_lease=None):
        if not isinstance(prepared, PreparedToolExecution):
            raise TypeError("prepared 必须是 PreparedToolExecution")
        if resource_lease is not None and not isinstance(resource_lease, ResourceLease):
            raise TypeError("resource_lease 必须为空或 ResourceLease")
        if resource_lease is not None:
            if resource_lease.manager is not self.lock_manager:
                raise ValueError("resource_lease 不属于当前 ToolExecutor 的锁管理器")
            if resource_lease.accesses != prepared.resources:
                raise ValueError("resource_lease 与 prepared.resources 不匹配")

        # lease：批量调度器传入或单独执行时阻塞获取的资源租约
        lease = resource_lease or self.lock_manager.acquire(
            f"direct:{uuid.uuid4().hex}",
            prepared.resources,
        )
        with lease:
            return self._execute_with_retry(prepared)

    # 执行已持有全部资源的工具，并按策略自动重试
    # prepared：已完成预处理且资源受保护的工具执行对象
    def _execute_with_retry(self, prepared):
        # name：当前已准备工具的注册名称
        name = prepared.name
        # args：当前已通过 Schema 校验的工具参数
        args = prepared.args
        # tool：当前已准备工具的注册定义
        tool = prepared.tool
        started_at = time.perf_counter()

        # 4. 按工具自己的 RetryPolicy 进入有限次数执行循环
        policy = tool.retry_policy
        for attempt in range(1, policy.max_attempts + 1):
            try:
                # 4. 使用模型生成的参数执行真正的工具函数
                value = tool.function(**args)

                # 5. 使用 Pydantic 校验并将业务返回值规范化为 JSON 可序列化数据
                try:
                    # validated_output：通过 Pydantic 返回模型校验的业务结果
                    validated_output = tool.output_model.model_validate(value)

                    # serialized_value：已转换为标准 JSON 类型的工具业务结果
                    serialized_value = validated_output.model_dump(mode="json")
                except ValidationError as error:
                    return ToolResult.failure(
                        name,
                        ErrorCode.INVALID_OUTPUT,
                        self._format_output_error(error),
                        duration_ms=self._elapsed_ms(started_at),
                        attempts=attempt,
                    )

                if attempt > 1:
                    logger.info("✅ 工具第 %d 次执行成功", attempt)
                return ToolResult.success(
                    name,
                    serialized_value,
                    duration_ms=self._elapsed_ms(started_at),
                    attempts=attempt,
                )
            except UnsafeRequestError as error:
                # 6. 安全拦截属于不可重试错误，立即返回
                return ToolResult.failure(
                    name,
                    ErrorCode.UNSAFE_REQUEST,
                    str(error),
                    duration_ms=self._elapsed_ms(started_at),
                    attempts=attempt,
                )
            except (ValueError, SyntaxError, ArithmeticError) as error:
                # 7. 工具发现参数语义错误时交给模型修正，不重复执行
                return ToolResult.failure(
                    name,
                    ErrorCode.INVALID_ARGUMENTS,
                    str(error),
                    duration_ms=self._elapsed_ms(started_at),
                    attempts=attempt,
                )
            except Exception as error:
                # 8. 将网络、超时、限流等基础设施异常转换为稳定错误码
                error_code, message = self._classify_exception(error)
                result = ToolResult.failure(
                    name,
                    error_code,
                    message,
                    duration_ms=self._elapsed_ms(started_at),
                    attempts=attempt,
                    retry_exhausted=(
                        ERROR_POLICIES[error_code][0]
                        and (not tool.idempotent or attempt >= policy.max_attempts)
                    ),
                )

                # 9. 只有可自动重试、幂等且仍有次数的工具才允许再次执行
                can_retry = (
                    result.auto_retryable
                    and tool.idempotent
                    and attempt < policy.max_attempts
                )
                if not can_retry:
                    return result

                # 10. 按指数退避和随机抖动计算等待时间，避免连续冲击服务
                delay = self._retry_delay(policy, attempt)
                logger.warning(
                    "⚠️ 工具第 %d 次执行失败：%s，%.2f 秒后重试",
                    attempt,
                    error_code.value,
                    delay,
                )
                self.sleeper(delay)

        raise RuntimeError("工具执行器进入了不可达状态")

    # 将 Pydantic 校验错误压缩为可安全返回的简短说明
    # error：Pydantic 返回模型校验失败异常
    @staticmethod
    def _format_output_error(error):
        # details：不包含原始输入和文档链接的结构化错误列表
        details = error.errors(include_url=False, include_input=False)

        # first_error：用于向 Agent 报告的第一个契约错误
        first_error = details[0]

        # location：发生输出契约错误的字段路径
        location = ".".join(str(part) for part in first_error.get("loc", ())) or "value"

        # message：Pydantic 生成的简短校验失败原因
        message = first_error.get("msg", "未知输出校验错误")
        return f"工具返回值不符合输出约定：{location}: {message}"

    # 将基础设施异常转换为稳定的错误码和安全说明
    # error：工具执行时抛出的原始异常
    @staticmethod
    def _classify_exception(error):
        # 按从具体到通用的顺序匹配，避免父类提前吞掉子类异常
        if isinstance(error, ToolAuthenticationError):
            return ErrorCode.AUTHENTICATION_ERROR, str(error)
        if isinstance(error, urllib.error.HTTPError):
            if error.code in {401, 403}:
                return ErrorCode.AUTHENTICATION_ERROR, f"工具服务认证失败，HTTP {error.code}"
            if error.code == 429:
                return ErrorCode.RATE_LIMITED, "工具服务请求过于频繁，HTTP 429"
            if 500 <= error.code <= 599:
                return ErrorCode.SERVER_ERROR, f"工具服务暂时不可用，HTTP {error.code}"
            return ErrorCode.INTERNAL_ERROR, f"工具服务返回未处理的 HTTP {error.code}"
        if isinstance(error, (TimeoutError, socket.timeout)):
            return ErrorCode.TIMEOUT, "工具请求超时"
        if isinstance(error, urllib.error.URLError):
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                return ErrorCode.TIMEOUT, "工具请求超时"
            return ErrorCode.NETWORK_ERROR, "工具网络连接失败"
        if isinstance(error, (socket.gaierror, ConnectionError)):
            return ErrorCode.NETWORK_ERROR, "工具网络连接失败"
        return ErrorCode.INTERNAL_ERROR, "工具执行过程中发生内部错误"

    # 计算指数退避和随机抖动后的等待时间
    # policy：工具重试策略
    # attempt：刚刚失败的执行次数
    def _retry_delay(self, policy, attempt):
        # 指数增长的等待时间不能超过工具配置的最大值
        exponential = min(policy.base_delay * 2 ** (attempt - 1), policy.max_delay)
        return exponential + self.jitter_fn(0, policy.jitter)

    # 计算工具执行总耗时
    # started_at：time.perf_counter 返回的开始时间
    @staticmethod
    def _elapsed_ms(started_at):
        return round((time.perf_counter() - started_at) * 1000)
