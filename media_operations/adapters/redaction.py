"""研究应用的审计/日志脱敏，不改变通用 Agent 源码。"""

import logging
import re


class Redactor:
    def __init__(self, secrets=()):
        self.secrets = sorted({value for value in secrets if isinstance(value, str) and len(value) >= 6}, key=len, reverse=True)

    def text(self, value):
        for secret in self.secrets:
            value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", value)
        value = re.sub(r"(?i)(Bearer\s+)[A-Za-z0-9._~-]+", r"\1[REDACTED]", value)
        value = re.sub(r"(?i)((?:api_key|access_token|password|secret|token)\s*[=:]\s*)[^\s&,;]+", r"\1[REDACTED]", value)
        return value

    def value(self, value):
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {str(key): "[REDACTED]" if any(word in str(key).lower() for word in ("password", "secret", "token", "api_key", "authorization", "cookie"))
                    else self.value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.value(item) for item in value]
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return f"<{type(value).__name__}>"


class ResearchLogFilter(logging.Filter):
    def __init__(self, redactor):
        super().__init__()
        self.redactor = redactor

    def filter(self, record):
        record.msg = self.redactor.text(record.getMessage())
        record.args = ()
        # 原始 traceback 可能包含供应商文本；工具契约已保存安全错误码。
        if record.exc_info or record.exc_text:
            record.exc_info, record.exc_text = None, None
        return True
