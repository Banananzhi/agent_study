import json
from dataclasses import dataclass
from enum import Enum


class ErrorCode(str, Enum):
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    RATE_LIMITED = "rate_limited"
    AUTHENTICATION_ERROR = "authentication_error"
    SERVER_ERROR = "server_error"
    UNSAFE_REQUEST = "unsafe_request"
    INTERNAL_ERROR = "internal_error"


ERROR_POLICIES = {
    ErrorCode.UNKNOWN_TOOL: (False, True),
    ErrorCode.INVALID_ARGUMENTS: (False, True),
    ErrorCode.TIMEOUT: (True, False),
    ErrorCode.NETWORK_ERROR: (True, False),
    ErrorCode.RATE_LIMITED: (True, False),
    ErrorCode.AUTHENTICATION_ERROR: (False, False),
    ErrorCode.SERVER_ERROR: (True, False),
    ErrorCode.UNSAFE_REQUEST: (False, False),
    ErrorCode.INTERNAL_ERROR: (False, False),
}


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    tool: str
    value: object | None = None
    error_code: ErrorCode | None = None
    error_message: str | None = None
    auto_retryable: bool = False
    model_recoverable: bool = False
    attempts: int = 0
    retry_exhausted: bool = False
    duration_ms: int = 0
    suggestions: tuple[str, ...] = ()

    # 校验工具结果内部字段是否保持一致
    def __post_init__(self):
        if type(self.ok) is not bool:
            raise TypeError("ok 必须是布尔值")
        if not isinstance(self.tool, str) or not self.tool.strip():
            raise ValueError("tool 不能为空")
        if type(self.duration_ms) is not int or self.duration_ms < 0:
            raise ValueError("duration_ms 必须是非负整数")
        if type(self.auto_retryable) is not bool or type(self.model_recoverable) is not bool:
            raise TypeError("恢复策略字段必须是布尔值")
        if type(self.attempts) is not int or self.attempts < 0:
            raise ValueError("attempts 必须是非负整数")
        if type(self.retry_exhausted) is not bool:
            raise TypeError("retry_exhausted 必须是布尔值")
        if not isinstance(self.suggestions, tuple):
            raise TypeError("suggestions 必须是元组")
        if len(self.suggestions) > 3:
            raise ValueError("suggestions 最多包含 3 个工具")
        if self.ok:
            if self.error_code is not None or self.error_message is not None:
                raise ValueError("成功结果不能包含错误信息")
            if self.auto_retryable or self.model_recoverable or self.suggestions:
                raise ValueError("成功结果不能包含恢复策略或工具建议信息")
            if self.attempts < 1 or self.retry_exhausted:
                raise ValueError("成功结果必须至少执行一次且不能标记重试耗尽")
        else:
            if not isinstance(self.error_code, ErrorCode):
                raise TypeError("失败结果必须包含 ErrorCode")
            if not isinstance(self.error_message, str) or not self.error_message.strip():
                raise ValueError("失败结果必须包含错误说明")
            if self.value is not None:
                raise ValueError("失败结果不能包含 value")
            if self.retry_exhausted and not self.auto_retryable:
                raise ValueError("只有可自动重试的错误才能标记重试耗尽")
            expected_policy = ERROR_POLICIES[self.error_code]
            actual_policy = (self.auto_retryable, self.model_recoverable)
            if actual_policy != expected_policy:
                raise ValueError("恢复策略必须与 ErrorCode 保持一致")

    # 创建工具执行成功的结果
    # tool：工具注册名称
    # value：工具返回的数据
    # duration_ms：工具执行耗时
    # attempts：工具实际执行次数
    @classmethod
    def success(cls, tool, value, duration_ms=0, attempts=1):
        return cls(ok=True, tool=tool, value=value, duration_ms=duration_ms, attempts=attempts)

    # 创建工具执行失败的结果
    # tool：工具注册名称
    # error_code：稳定的错误类型
    # error_message：可安全返回给模型的错误说明
    # duration_ms：工具执行耗时
    # suggestions：少量相关工具建议
    # attempts：工具实际执行次数
    # retry_exhausted：执行器是否已经用完重试机会
    @classmethod
    def failure(
        cls,
        tool,
        error_code,
        error_message,
        duration_ms=0,
        suggestions=(),
        attempts=0,
        retry_exhausted=False,
    ):
        auto_retryable, model_recoverable = ERROR_POLICIES[error_code]
        return cls(
            ok=False,
            tool=tool,
            error_code=error_code,
            error_message=error_message,
            auto_retryable=auto_retryable,
            model_recoverable=model_recoverable,
            attempts=attempts,
            retry_exhausted=retry_exhausted,
            duration_ms=duration_ms,
            suggestions=tuple(suggestions),
        )

    # 转换为适合放入 Observation 的字典
    def to_dict(self):
        result = {
            "ok": self.ok,
            "tool": self.tool,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
        }
        if self.ok:
            result["value"] = self.value
            return result
        result["error"] = {
            "code": self.error_code.value,
            "message": self.error_message,
            "auto_retryable": self.auto_retryable,
            "model_recoverable": self.model_recoverable,
            "retry_exhausted": self.retry_exhausted,
        }
        if self.suggestions:
            result["error"]["suggestions"] = list(self.suggestions)
        return result

    # 序列化为可直接返回模型的 JSON Observation
    def to_observation(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)
