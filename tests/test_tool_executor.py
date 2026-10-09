import asyncio
import json
import unittest
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from mcp.shared.exceptions import MCPError as SDKMCPError
from pydantic import ConfigDict, RootModel

from integrations.mcp import (
    MCPAuthenticationError,
    MCPClientManager,
    MCPConnectionError,
    MCPInvalidArgumentsError,
    MCPProtocolError,
    MCPRemoteExecutionError,
    MCPRateLimitError,
    MCPServerConfig,
    MCPServerError,
    MCPTimeoutError,
    MCPToolAdapter,
)
from tooling.executor import ToolExecutor
from tooling.policy import PolicyAction, SideEffectLevel, ToolExecutionPolicy
from tooling.registry import (
    CREATE_FILE_SCHEMA,
    READ_FILE_SCHEMA,
    WRITE_FILE_SCHEMA,
    RetryPolicy,
    Tool,
    ToolAuthenticationError,
    UnsafeRequestError,
    TOOLS,
    create_file,
    read_file,
    write_file,
)
from tooling.resources import AccessMode, ResourceAccess, ResourceLockManager
from tooling.result import ErrorCode
from tooling.scheduler import BatchToolCall, ToolBatchExecutor


SCHEMA = {
    "type": "function",
    "function": {
        "name": "fake",
        "description": "测试工具",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}

class FakeOutput(RootModel):
    """返回测试字符串。"""

    model_config = ConfigDict(strict=True)
    root: str


# 创建带重试策略的测试工具
# function：测试工具函数
# max_attempts：最大执行次数
# idempotent：是否允许重复执行
# output_model：测试工具成功返回值的 Pydantic 模型
# side_effect_level：测试工具声明的副作用等级
def make_tool(
    function,
    max_attempts=3,
    idempotent=True,
    output_model=None,
    resource_resolver=None,
    side_effect_level=SideEffectLevel.NONE,
):
    return Tool(
        function=function,
        schema=SCHEMA,
        display_name="Fake",
        output_model=output_model or FakeOutput,
        resource_resolver=resource_resolver,
        side_effect_level=side_effect_level,
        retry_policy=RetryPolicy(
            max_attempts=max_attempts,
            base_delay=0.5,
            max_delay=2.0,
            jitter=0.2,
        ),
        idempotent=idempotent,
    )


class ToolExecutorTests(unittest.TestCase):
    # 验证底层 SDK、HTTP 与传输异常会在 MCP 边界完成准确分类
    def test_mcp_client_exception_classification(self):
        # request：用于构造不包含敏感数据的 HTTP 状态异常
        request = httpx.Request("POST", "https://example.com/mcp")
        # unauthorized_response：模拟 MCP HTTP 传输认证失败
        unauthorized_response = httpx.Response(401, request=request)
        # rate_limited_response：模拟 MCP HTTP 传输限流
        rate_limited_response = httpx.Response(429, request=request)
        # server_response：模拟 MCP Server 暂时不可用
        server_response = httpx.Response(503, request=request)
        # cases：底层异常与 MCP 适配层预期异常类型的映射
        cases = [
            (SDKMCPError(-32602, "Invalid params"), MCPInvalidArgumentsError),
            (SDKMCPError(-32600, "Invalid request"), MCPProtocolError),
            (SDKMCPError(-32603, "Internal error"), MCPServerError),
            (httpx.ReadTimeout("timed out", request=request), MCPTimeoutError),
            (httpx.ConnectError("connect failed", request=request), MCPConnectionError),
            (
                httpx.HTTPStatusError(
                    "unauthorized",
                    request=request,
                    response=unauthorized_response,
                ),
                MCPAuthenticationError,
            ),
            (
                httpx.HTTPStatusError(
                    "rate limited",
                    request=request,
                    response=rate_limited_response,
                ),
                MCPRateLimitError,
            ),
            (
                httpx.HTTPStatusError(
                    "server unavailable",
                    request=request,
                    response=server_response,
                ),
                MCPServerError,
            ),
        ]

        # source_error：MCP Client 或底层 HTTP 库抛出的原始异常
        # expected_type：适配层应生成的稳定 MCP 异常类型
        for source_error, expected_type in cases:
            with self.subTest(source_type=type(source_error).__name__):
                # classified：MCP 边界归一化后的异常
                classified = MCPClientManager._classify_client_exception(
                    source_error,
                    "工具调用",
                )
                self.assertIsInstance(classified, expected_type)

    # 验证 CallToolResult.is_error 会被识别成可交给模型修正的远程业务错误
    def test_mcp_error_result_becomes_remote_execution_error(self):
        # client：模拟已连接且返回 MCP 错误结果的异步 Client
        client = SimpleNamespace()
        client.call_tool = Mock()

        # call_tool：返回带文本错误块的模拟 MCP 调用结果
        async def call_tool(tool_name, arguments, raise_on_error):
            return SimpleNamespace(
                is_error=True,
                content=[SimpleNamespace(text="仓库名称格式无效")],
            )

        client.call_tool = call_tool
        # manager：跳过线程和网络初始化，仅测试异步结果规范化边界
        manager = object.__new__(MCPClientManager)
        manager.clients = {"deepwiki": client}

        with self.assertRaises(MCPRemoteExecutionError) as raised:
            asyncio.run(
                manager._call_tool(
                    "deepwiki",
                    "read_wiki_structure",
                    {"repoName": "invalid"},
                )
            )

        self.assertEqual(raised.exception.error_code, ErrorCode.REMOTE_ERROR)
        self.assertIn("仓库名称格式无效", str(raised.exception))

    # 验证六类 MCP 错误会转换成正确的稳定错误码与恢复策略
    def test_mcp_errors_follow_declared_recovery_policies(self):
        # cases：MCP 适配层异常及其预期 ToolResult 路由语义
        cases = [
            (
                MCPInvalidArgumentsError("参数字段无效"),
                ErrorCode.INVALID_ARGUMENTS,
                False,
                True,
            ),
            (
                MCPConnectionError("连接失败"),
                ErrorCode.NETWORK_ERROR,
                True,
                False,
            ),
            (
                MCPTimeoutError("调用超时"),
                ErrorCode.TIMEOUT,
                True,
                False,
            ),
            (
                MCPAuthenticationError("认证失败"),
                ErrorCode.AUTHENTICATION_ERROR,
                False,
                False,
            ),
            (
                MCPRemoteExecutionError("仓库名称不存在"),
                ErrorCode.REMOTE_ERROR,
                False,
                True,
            ),
            (
                MCPProtocolError("响应不符合 MCP 协议"),
                ErrorCode.PROTOCOL_ERROR,
                False,
                False,
            ),
        ]

        # error：当前模拟的 MCP 已分类异常
        # expected_code：异常应转换成的稳定错误码
        # auto_retryable：程序是否可以自动重试幂等工具
        # model_recoverable：是否应将错误返回模型修正调用
        for error, expected_code, auto_retryable, model_recoverable in cases:
            with self.subTest(error_type=type(error).__name__):
                # failing_tool：抛出当前 MCP 异常的单次测试工具
                def failing_tool(current_error=error):
                    raise current_error

                # result：限制为一次尝试后生成的统一工具失败结果
                result = ToolExecutor({
                    "fake": make_tool(failing_tool, max_attempts=1)
                }).execute("fake", {})

                self.assertEqual(result.error_code, expected_code)
                self.assertEqual(result.auto_retryable, auto_retryable)
                self.assertEqual(result.model_recoverable, model_recoverable)
                self.assertEqual(result.retry_exhausted, auto_retryable)
                self.assertEqual(result.attempts, 1)

    # 验证 MCP 连接错误仅在工具幂等且仍有次数时由程序自动重试
    def test_mcp_connection_error_retries_idempotent_tool(self):
        # calls：记录远程 MCP 代理函数的实际调用次数
        calls = []
        # delays：记录指数退避产生的等待时间
        delays = []

        # flaky_mcp_tool：第一次连接失败，第二次恢复成功的只读 MCP 工具
        def flaky_mcp_tool():
            calls.append(1)
            if len(calls) == 1:
                raise MCPConnectionError("MCP 网络连接失败")
            return "ok"

        # executor：关闭随机抖动以验证稳定的重试行为
        executor = ToolExecutor(
            {"fake": make_tool(flaky_mcp_tool, max_attempts=2)},
            sleeper=delays.append,
            jitter_fn=lambda start, end: 0,
        )
        # result：第二次 MCP 调用成功后的统一结果
        result = executor.execute("fake", {})

        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(delays, [0.5])

    # 验证 MCP Tool 会增加 Server 命名空间并映射只读 annotations
    def test_mcp_tool_adapter_builds_namespaced_read_only_tool(self):
        # manager：记录适配后代理函数发出的远程工具调用
        manager = SimpleNamespace()
        manager.call_tool = Mock(return_value={"answer": "ok"})
        # config：已知只读远程 Server 的连接和默认安全配置
        config = MCPServerConfig(
            name="docs-server",
            url="https://example.com/mcp",
            default_side_effect_level=SideEffectLevel.EXTERNAL_WRITE,
        )
        # remote_tool：模拟 FastMCP list_tools 返回的只读工具定义
        remote_tool = SimpleNamespace(
            name="search-docs",
            description="搜索远程文档",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            annotations=SimpleNamespace(
                read_only_hint=True,
                destructive_hint=False,
                idempotent_hint=True,
            ),
        )

        # tool：完成 Schema、命名空间和安全元数据转换的项目工具
        tool = MCPToolAdapter(manager, config).adapt(remote_tool)
        # result：通过现有 ToolExecutor 执行远程代理后的统一结果
        result = ToolExecutor({tool.name: tool}).execute(
            tool.name,
            {"query": "MCP"},
        )

        self.assertEqual(tool.name, "docs_server__search_docs")
        self.assertEqual(tool.side_effect_level, SideEffectLevel.NONE)
        self.assertTrue(tool.idempotent)
        self.assertTrue(result.ok)
        self.assertEqual(result.value, {"answer": "ok"})
        manager.call_tool.assert_called_once_with(
            "docs-server",
            "search-docs",
            {"query": "MCP"},
        )

    # 验证缺少 annotations 时使用 Server 配置的保守副作用等级
    def test_mcp_tool_adapter_uses_server_default_side_effect(self):
        # manager：本测试只构建工具，不实际发起远程调用
        manager = SimpleNamespace(call_tool=lambda *args: "ok")
        # remote_tool：未声明任何安全 annotations 的远程工具
        remote_tool = SimpleNamespace(
            name="submit",
            description="提交业务数据",
            input_schema={"type": "object", "properties": {}},
            annotations=None,
        )
        # config：未知工具默认按照外部写入处理的 Server 配置
        config = MCPServerConfig(
            name="business",
            url="https://example.com/mcp",
            default_side_effect_level=SideEffectLevel.EXTERNAL_WRITE,
        )

        # tool：使用 Server 默认等级完成适配的项目工具
        tool = MCPToolAdapter(manager, config).adapt(remote_tool)

        self.assertEqual(tool.side_effect_level, SideEffectLevel.EXTERNAL_WRITE)
        self.assertFalse(tool.idempotent)

    # 验证现有工具注册了与行为一致的副作用等级
    def test_registered_tool_side_effect_levels(self):
        self.assertEqual(TOOLS["calculator"].side_effect_level, SideEffectLevel.NONE)
        self.assertEqual(TOOLS["read_file"].side_effect_level, SideEffectLevel.NONE)
        self.assertEqual(
            TOOLS["create_file"].side_effect_level,
            SideEffectLevel.LOCAL_WRITE,
        )
        self.assertEqual(
            TOOLS["write_file"].side_effect_level,
            SideEffectLevel.LOCAL_WRITE,
        )

    # 验证外部写入工具在资源解析和实际执行前返回需要审批
    def test_external_write_requires_approval_before_execution(self):
        # events：记录资源解析器和工具函数是否被意外调用
        events = []

        # external_write_tool：模拟会改变外部系统状态的工具函数
        def external_write_tool():
            events.append("executed")
            return "ok"

        # resolve_resources：模拟副作用工具的资源解析器
        # args：已经通过输入 Schema 校验的空参数字典
        def resolve_resources(args):
            events.append("resources_resolved")
            return ()

        # executor：使用默认副作用策略的工具执行器
        executor = ToolExecutor({
            "fake": make_tool(
                external_write_tool,
                resource_resolver=resolve_resources,
                side_effect_level=SideEffectLevel.EXTERNAL_WRITE,
            )
        })
        # result：应在执行前生成的稳定审批错误结果
        result = executor.execute("fake", {})

        self.assertEqual(result.error_code, ErrorCode.APPROVAL_REQUIRED)
        self.assertEqual(result.attempts, 0)
        self.assertFalse(result.auto_retryable)
        self.assertFalse(result.model_recoverable)
        self.assertEqual(events, [])

    # 验证调用方可以通过统一策略明确禁止某个副作用等级
    def test_execution_policy_can_deny_side_effect_level(self):
        # calls：记录被策略禁止的工具是否实际执行
        calls = []
        # policy：将默认允许的本地写入覆盖为明确禁止
        policy = ToolExecutionPolicy({
            SideEffectLevel.LOCAL_WRITE: PolicyAction.DENY,
        })
        # executor：注入自定义副作用策略的工具执行器
        executor = ToolExecutor(
            {
                "fake": make_tool(
                    lambda: calls.append(1) or "ok",
                    side_effect_level=SideEffectLevel.LOCAL_WRITE,
                )
            },
            execution_policy=policy,
        )
        # result：工具尚未执行时返回的策略拒绝结果
        result = executor.execute("fake", {})

        self.assertEqual(result.error_code, ErrorCode.POLICY_DENIED)
        self.assertEqual(result.attempts, 0)
        self.assertEqual(calls, [])

    # 验证临时超时后能够自动重试成功
    def test_retry_until_success(self):
        calls = []
        delays = []

        # 前两次调用超时，第三次返回成功
        def flaky_tool():
            calls.append(1)
            if len(calls) < 3:
                raise TimeoutError
            return "ok"

        executor = ToolExecutor(
            {"fake": make_tool(flaky_tool)},
            sleeper=delays.append,
            jitter_fn=lambda start, end: 0,
        )
        result = executor.execute("fake", {})

        self.assertTrue(result.ok)
        self.assertEqual(result.value, "ok")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(delays, [0.5, 1.0])

    # 验证重试耗尽后返回最终失败
    def test_retry_exhausted(self):
        calls = []
        delays = []

        # 每次调用都模拟超时
        def timeout_tool():
            calls.append(1)
            raise TimeoutError

        executor = ToolExecutor(
            {"fake": make_tool(timeout_tool)},
            sleeper=delays.append,
            jitter_fn=lambda start, end: 0,
        )
        result = executor.execute("fake", {})

        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ErrorCode.TIMEOUT)
        self.assertEqual(result.attempts, 3)
        self.assertTrue(result.retry_exhausted)
        self.assertEqual(len(calls), 3)
        self.assertEqual(delays, [0.5, 1.0])

    # 验证认证错误不会自动重试
    def test_authentication_error_does_not_retry(self):
        calls = []

        # 模拟工具服务认证失败
        def auth_error_tool():
            calls.append(1)
            raise ToolAuthenticationError("认证失败")

        executor = ToolExecutor(
            {"fake": make_tool(auth_error_tool)},
            sleeper=lambda delay: self.fail("认证错误不应等待重试"),
        )
        result = executor.execute("fake", {})

        self.assertEqual(result.error_code, ErrorCode.AUTHENTICATION_ERROR)
        self.assertEqual(result.attempts, 1)
        self.assertFalse(result.retry_exhausted)
        self.assertEqual(len(calls), 1)

    # 验证非幂等工具即使超时也不会自动重试
    def test_non_idempotent_tool_does_not_retry(self):
        calls = []

        # 模拟具有副作用的工具超时
        def side_effect_tool():
            calls.append(1)
            raise TimeoutError

        executor = ToolExecutor(
            {"fake": make_tool(side_effect_tool, idempotent=False)},
            sleeper=lambda delay: self.fail("非幂等工具不应等待重试"),
        )
        result = executor.execute("fake", {})

        self.assertEqual(result.error_code, ErrorCode.TIMEOUT)
        self.assertEqual(result.attempts, 1)
        self.assertTrue(result.retry_exhausted)
        self.assertEqual(len(calls), 1)

    # 验证参数校验失败时不会调用工具或进入重试
    def test_invalid_arguments_do_not_execute(self):
        called = []
        schema = {
            "type": "function",
            "function": {
                "name": "fake",
                "description": "测试工具",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
        tool = Tool(
            function=lambda value: called.append(value),
            schema=schema,
            display_name="Fake",
            output_model=FakeOutput,
        )
        result = ToolExecutor({"fake": tool}).execute("fake", {})

        self.assertEqual(result.error_code, ErrorCode.INVALID_ARGUMENTS)
        self.assertEqual(result.attempts, 0)
        self.assertEqual(called, [])

    # 验证工具未报错但返回类型不符合约定时会被拦截
    def test_invalid_output_is_rejected_without_retry(self):
        # calls：记录工具函数的实际执行次数
        calls = []

        # invalid_output_tool：声明返回字符串却实际返回整数的测试工具
        def invalid_output_tool():
            calls.append(1)
            return 123

        # executor：配置了多次尝试但不应对输出契约错误重试的执行器
        executor = ToolExecutor(
            {"fake": make_tool(invalid_output_tool, max_attempts=3)},
            sleeper=lambda delay: self.fail("错误输出不应等待重试"),
        )

        # result：工具业务返回值与 Output Schema 不一致时的统一失败结果
        result = executor.execute("fake", {})

        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ErrorCode.INVALID_OUTPUT)
        self.assertEqual(result.attempts, 1)
        self.assertFalse(result.auto_retryable)
        self.assertFalse(result.model_recoverable)
        self.assertEqual(len(calls), 1)


class ToolBatchExecutorTests(unittest.TestCase):
    # 验证不同 ToolExecutor 默认共享同一个进程级资源锁管理器
    def test_tool_executors_share_default_resource_lock_manager(self):
        # first_executor：模拟第一个 Agent 持有的工具执行器
        first_executor = ToolExecutor()
        # second_executor：模拟同一进程内另一个 Agent 持有的工具执行器
        second_executor = ToolExecutor()
        # write_access：两个执行器竞争的同一文件写资源
        write_access = (
            ResourceAccess("file", "C:/workspace/shared.txt", AccessMode.WRITE),
        )
        # first_lease：第一个执行器成功申请的文件写租约
        first_lease = first_executor.lock_manager.try_acquire(
            "first_executor",
            write_access,
        )

        self.assertIs(first_executor.lock_manager, second_executor.lock_manager)
        self.assertIsNotNone(first_lease)
        try:
            self.assertIsNone(
                second_executor.lock_manager.try_acquire(
                    "second_executor",
                    write_access,
                )
            )
        finally:
            first_lease.release()

    # 验证同资源等待调用不占用 Worker，后续无冲突调用仍可立即执行
    def test_waiting_lock_does_not_occupy_worker(self):
        # first_started：第一个独占资源工具已进入 Worker 的事件
        first_started = threading.Event()
        # release_first：测试主线程允许第一个工具释放资源的事件
        release_first = threading.Event()
        # second_started：同资源后续工具已开始执行的事件
        second_started = threading.Event()
        # independent_started：无冲突工具已进入另一个 Worker 的事件
        independent_started = threading.Event()

        # 占用 file:a 直到测试主线程允许结束
        def first_tool():
            first_started.set()
            if not release_first.wait(1):
                raise TimeoutError("测试未及时释放第一个工具")
            return "first"

        # 只有在 file:a 被释放后才应开始的后续工具
        def second_tool():
            second_started.set()
            return "second"

        # 不使用 file:a，应在第一个工具阻塞时仍立即执行
        def independent_tool():
            independent_started.set()
            return "independent"

        # 为两个测试工具返回相同的 file:a 写资源
        # args：已通过测试 Schema 校验的空参数字典
        def same_file_write(args):
            return (
                ResourceAccess("file", "C:/workspace/a.txt", AccessMode.WRITE),
            )
        # registry：包含两个冲突工具和一个无资源工具的测试注册表
        registry = {
            "first": make_tool(first_tool, resource_resolver=same_file_write),
            "second": make_tool(second_tool, resource_resolver=same_file_write),
            "independent": make_tool(independent_tool),
        }
        # scheduler：只提供两个 Worker 的测试批量调度器
        scheduler = ToolBatchExecutor(ToolExecutor(registry), max_parallel_tools=2)
        # calls：刻意将被锁阻塞的 second 放在无冲突 independent 之前
        calls = [
            BatchToolCall(0, "call_1", "first", {}),
            BatchToolCall(1, "call_2", "second", {}),
            BatchToolCall(2, "call_3", "independent", {}),
        ]
        # result_holder：从后台调度线程取回有序 ToolResult 列表的容器
        result_holder = {}

        # 在后台线程执行批次，便于测试主线程控制第一个工具的释放时机
        def run_batch():
            result_holder["results"] = scheduler.execute_batch(calls)

        # batch_thread：执行完整批量调度循环的后台线程
        batch_thread = threading.Thread(target=run_batch)
        batch_thread.start()
        self.assertTrue(first_started.wait(1))
        self.assertTrue(independent_started.wait(1))
        self.assertFalse(second_started.is_set())

        release_first.set()
        batch_thread.join(1)
        self.assertFalse(batch_thread.is_alive())
        self.assertTrue(second_started.is_set())
        self.assertEqual(
            [result.value for result in result_holder["results"]],
            ["first", "second", "independent"],
        )

    # 验证资源锁允许同文件多个读租约，但会拒绝同文件写租约
    def test_resource_lock_supports_shared_reads(self):
        # manager：用于验证读写冲突规则的资源锁管理器
        manager = ResourceLockManager()
        # read_access：对同一文件的只读访问声明
        read_access = (ResourceAccess("file", "C:/workspace/a.txt", AccessMode.READ),)
        # write_access：对同一文件的写入访问声明
        write_access = (ResourceAccess("file", "C:/workspace/a.txt", AccessMode.WRITE),)

        # first_read：第一个成功获得的共享读租约
        first_read = manager.try_acquire("read_1", read_access)
        # second_read：与第一个租约共享同一文件的读租约
        second_read = manager.try_acquire("read_2", read_access)
        self.assertIsNotNone(first_read)
        self.assertIsNotNone(second_read)
        self.assertIsNone(manager.try_acquire("write_1", write_access))

        first_read.release()
        second_read.release()
        # write_lease：全部读租约释放后才能获得的写租约
        write_lease = manager.try_acquire("write_2", write_access)
        self.assertIsNotNone(write_lease)
        write_lease.release()

    # 验证同一文件的创建和后续分段写入会按模型原始顺序执行
    def test_same_file_batch_preserves_write_order(self):
        with TemporaryDirectory() as temporary_directory:
            # workspace_root：与真实项目隔离的临时工作区
            workspace_root = Path(temporary_directory)
            # scheduler：使用真实文件工具和资源解析器的批量调度器
            scheduler = ToolBatchExecutor(ToolExecutor(), max_parallel_tools=2)
            # calls：依次创建文件并从首段 next_offset 继续追加的调用
            calls = [
                BatchToolCall(
                    0,
                    "call_1",
                    "create_file",
                    {"path": "batch.txt", "content": "abc"},
                ),
                BatchToolCall(
                    1,
                    "call_2",
                    "write_file",
                    {"path": "batch.txt", "content": "def", "offset": 3},
                ),
            ]
            with patch("tooling.registry.WORKSPACE_ROOT", workspace_root):
                # results：按原始调用顺序返回的文件工具结果
                results = scheduler.execute_batch(calls)

            self.assertTrue(all(result.ok for result in results))
            self.assertEqual(
                (workspace_root / "batch.txt").read_text(encoding="utf-8"),
                "abcdef",
            )


class FileToolTests(unittest.TestCase):
    # 验证创建工具可以建立父目录，写入工具可以替换已有内容
    def test_create_and_write_file(self):
        with TemporaryDirectory() as temporary_directory:
            # workspace_root：与真实项目隔离的临时工作区
            workspace_root = Path(temporary_directory)
            with patch("tooling.registry.WORKSPACE_ROOT", workspace_root):
                # created：创建文件工具返回的标准业务结果
                created = create_file("notes/result.txt", "初始内容")
                # written：覆盖写入工具返回的标准业务结果
                written = write_file("notes/result.txt", "更新内容")

            self.assertEqual(
                created,
                {"path": "notes/result.txt", "chars_written": 4, "next_offset": 4},
            )
            self.assertEqual(
                written,
                {"path": "notes/result.txt", "chars_written": 4, "next_offset": 4},
            )
            self.assertEqual((workspace_root / "notes/result.txt").read_text(encoding="utf-8"), "更新内容")

    # 验证创建工具不会覆盖同名文件
    def test_create_file_rejects_existing_file(self):
        with TemporaryDirectory() as temporary_directory:
            # workspace_root：与真实项目隔离的临时工作区
            workspace_root = Path(temporary_directory)
            (workspace_root / "existing.txt").write_text("原内容", encoding="utf-8")
            with patch("tooling.registry.WORKSPACE_ROOT", workspace_root):
                with self.assertRaises(ValueError):
                    create_file("existing.txt", "新内容")

            self.assertEqual((workspace_root / "existing.txt").read_text(encoding="utf-8"), "原内容")

    # 验证文件工具会拒绝使用 .. 越出工作区
    def test_file_tool_rejects_path_traversal(self):
        with TemporaryDirectory() as temporary_directory:
            # workspace_root：用于路径边界校验的临时工作区
            workspace_root = Path(temporary_directory)
            with patch("tooling.registry.WORKSPACE_ROOT", workspace_root):
                with self.assertRaises(UnsafeRequestError):
                    create_file("../outside.txt", "越界内容")

    # 验证两个文件工具的 Function Calling 参数约定
    def test_file_tool_schemas(self):
        self.assertEqual(CREATE_FILE_SCHEMA["function"]["parameters"]["required"], ["path"])
        self.assertEqual(
            WRITE_FILE_SCHEMA["function"]["parameters"]["required"],
            ["path", "content"],
        )
        self.assertEqual(READ_FILE_SCHEMA["function"]["parameters"]["required"], ["path"])

    # 验证读取工具可以根据 next_offset 连续读取全部分页
    def test_read_file_with_pagination(self):
        with TemporaryDirectory() as temporary_directory:
            # workspace_root：与真实项目隔离的临时工作区
            workspace_root = Path(temporary_directory)
            (workspace_root / "content.txt").write_text("一二三四五", encoding="utf-8")
            with patch("tooling.registry.WORKSPACE_ROOT", workspace_root):
                # first_page：从字符下标 0 开始读取的第一页
                first_page = read_file("content.txt", offset=0, max_chars=3)
                # second_page：使用上一页 next_offset 读取的后续内容
                second_page = read_file(
                    "content.txt",
                    offset=first_page["next_offset"],
                    max_chars=3,
                )

            self.assertEqual(
                first_page,
                {
                    "path": "content.txt",
                    "offset": 0,
                    "next_offset": 3,
                    "has_more": True,
                    "content": "一二三",
                },
            )
            self.assertEqual(second_page["content"], "四五")
            self.assertEqual(second_page["next_offset"], 5)
            self.assertFalse(second_page["has_more"])
            self.assertEqual(first_page["content"] + second_page["content"], "一二三四五")

    # 验证读取完整文件时不会误报截断
    def test_read_complete_file(self):
        with TemporaryDirectory() as temporary_directory:
            # workspace_root：与真实项目隔离的临时工作区
            workspace_root = Path(temporary_directory)
            (workspace_root / "empty.txt").write_text("", encoding="utf-8")
            with patch("tooling.registry.WORKSPACE_ROOT", workspace_root):
                # result：空文件的完整读取结果
                result = read_file("empty.txt")

            self.assertEqual(
                result,
                {
                    "path": "empty.txt",
                    "offset": 0,
                    "next_offset": 0,
                    "has_more": False,
                    "content": "",
                },
            )

    # 验证写入工具可以根据 next_offset 按顺序追加多个分段
    def test_write_file_with_offsets(self):
        with TemporaryDirectory() as temporary_directory:
            # workspace_root：与真实项目隔离的临时工作区
            workspace_root = Path(temporary_directory)
            with patch("tooling.registry.WORKSPACE_ROOT", workspace_root):
                # first_chunk：创建文件时写入的首个文本分段
                first_chunk = create_file("chunks.txt", "第一")
                # second_chunk：根据首段返回偏移追加的第二个文本分段
                second_chunk = write_file(
                    "chunks.txt",
                    "第二",
                    offset=first_chunk["next_offset"],
                )
                # third_chunk：根据第二段返回偏移追加的第三个文本分段
                third_chunk = write_file(
                    "chunks.txt",
                    "第三",
                    offset=second_chunk["next_offset"],
                )

            self.assertEqual(third_chunk["next_offset"], 6)
            self.assertEqual(
                (workspace_root / "chunks.txt").read_text(encoding="utf-8"),
                "第一第二第三",
            )

    # 验证写入偏移与当前文件长度不一致时不会写入内容
    def test_write_file_rejects_wrong_offset(self):
        with TemporaryDirectory() as temporary_directory:
            # workspace_root：与真实项目隔离的临时工作区
            workspace_root = Path(temporary_directory)
            (workspace_root / "chunks.txt").write_text("已有内容", encoding="utf-8")
            with patch("tooling.registry.WORKSPACE_ROOT", workspace_root):
                with self.assertRaises(ValueError):
                    write_file("chunks.txt", "错误分段", offset=2)

            self.assertEqual(
                (workspace_root / "chunks.txt").read_text(encoding="utf-8"),
                "已有内容",
            )

    # 验证 JSON 转义会膨胀的文件内容不会被 Observation 二次截断
    def test_read_page_fits_observation_limit(self):
        with TemporaryDirectory() as temporary_directory:
            # workspace_root：与真实项目隔离的临时工作区
            workspace_root = Path(temporary_directory)
            (workspace_root / "lines.txt").write_text("\n" * 6000, encoding="utf-8")
            with patch("tooling.registry.WORKSPACE_ROOT", workspace_root):
                # result：经过工具执行器和 Pydantic 返回值校验的分页结果
                result = ToolExecutor().execute(
                    "read_file",
                    {"path": "lines.txt", "offset": 0, "max_chars": 6000},
                )

            # observation：将完整分页结果放入默认长度上限后的模型消息
            observation = result.to_observation(max_chars=8000)
            # payload：从 Observation 中恢复的结构化工具结果
            payload = json.loads(observation)

            self.assertNotIn("truncation", payload)
            self.assertTrue(payload["value"]["has_more"])
            self.assertEqual(payload["value"]["next_offset"], 3000)


if __name__ == "__main__":
    unittest.main()
