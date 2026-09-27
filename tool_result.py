import json
from dataclasses import dataclass
from enum import Enum


class ErrorCode(str, Enum):
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    INVALID_OUTPUT = "invalid_output"
    REPEATED_ACTION = "repeated_action"
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
    ErrorCode.INVALID_OUTPUT: (False, False),
    ErrorCode.REPEATED_ACTION: (False, True),
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

    # 将观察结果序列化为紧凑的 JSON 文本
    # payload：待序列化的观察结果字典
    @staticmethod
    def _serialize_observation(payload):
        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )

    # 构造极端情况下仍能放入长度上限的最小观察结果
    # max_chars：观察结果允许的最大字符数
    # original_chars：未截断观察结果的字符数
    def _minimal_observation(self, max_chars, original_chars):
        # tool_name：为最小观察结果保留的有限长度工具名称
        tool_name = self.tool[:64]

        # payload：丢弃非必要长文本后的最小观察结果结构
        payload = {
            "ok": self.ok,
            "tool": tool_name,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
            "truncation": {
                "truncated": True,
                "field": "observation",
                "original_observation_chars": original_chars,
                "returned_field_chars": 0,
            },
        }
        if self.ok:
            payload["value"] = ""
        else:
            payload["error"] = {
                "code": self.error_code.value,
                "message": "",
                "auto_retryable": self.auto_retryable,
                "model_recoverable": self.model_recoverable,
                "retry_exhausted": self.retry_exhausted,
            }

        # observation：序列化后的最小观察结果
        observation = self._serialize_observation(payload)
        if len(observation) > max_chars:
            raise ValueError("max_chars 太小，无法容纳最小 Observation 结构")
        return observation

    # 序列化为可直接返模型的 JSON Observation
    # max_chars：最大字符数，为空时不限制长度
    def to_observation(self, max_chars=None):
        if max_chars is not None and (type(max_chars) is not int or max_chars < 512):
            raise ValueError("max_chars 必须为空或大于等于 512 的整数")

        # payload：即将返回模型的结构化观察结果
        payload = self.to_dict()

        # observation：未经截断的完整观察结果 JSON
        observation = self._serialize_observation(payload)
        if max_chars is None or len(observation) <= max_chars:
            return observation

        # original_observation_chars：完整观察结果的原始字符数
        original_observation_chars = len(observation)

        if self.ok:
            # original_value：工具成功时返回的原始数据
            original_value = payload["value"]

            # source_text：用于安全截断的文本形式工具结果
            source_text = (
                original_value
                if isinstance(original_value, str)
                else self._serialize_observation(original_value)
            )

            # value_format：截断前工具结果的表达格式
            value_format = "text" if isinstance(original_value, str) else "json_text"

            # target：承载待截断字段的字典
            target = payload

            # field：待截断的成功结果字段名称
            field = "value"
        else:
            # source_text：用于安全截断的工具错误说明
            source_text = payload["error"]["message"]

            # value_format：错误说明始终以普通文本表达
            value_format = "text"

            # target：承载待截断错误说明的字典
            target = payload["error"]

            # field：待截断的错误说明字段名称
            field = "message"

        # truncation：告知模型截断位置和原始长度的元数据
        truncation = {
            "truncated": True,
            "field": field,
            "format": value_format,
            "original_observation_chars": original_observation_chars,
            "original_field_chars": len(source_text),
            "returned_field_chars": 0,
        }
        payload["truncation"] = truncation

        # low：二分查找中当前可保留字符数的下界
        low = 0

        # high：二分查找中当前可保留字符数的上界
        high = len(source_text)

        # best_observation：当前找到的最长且不超限观察结果
        best_observation = None
        while low <= high:
            # kept_chars：本次尝试保留的原始字段字符数
            kept_chars = (low + high) // 2

            # truncated_text：本次尝试放入 Observation 的截断文本
            truncated_text = source_text[:kept_chars] + "…"
            target[field] = truncated_text
            truncation["returned_field_chars"] = len(truncated_text)

            # candidate：本次尝试的完整 JSON Observation
            candidate = self._serialize_observation(payload)
            if len(candidate) <= max_chars:
                best_observation = candidate
                low = kept_chars + 1
            else:
                high = kept_chars - 1

        if best_observation is None:
            return self._minimal_observation(max_chars, original_observation_chars)
        return best_observation
