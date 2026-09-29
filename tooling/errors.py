from tooling.result import ErrorCode


class ClassifiedToolError(RuntimeError):
    # 初始化已经由工具适配层完成分类的异常
    # error_code：ToolExecutor 应转换成的稳定错误码
    # message：可安全写入 ToolResult 并返回模型的错误说明
    def __init__(self, error_code, message):
        if not isinstance(error_code, ErrorCode):
            raise TypeError("error_code 必须是 ErrorCode")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message 不能为空")
        self.error_code = error_code
        super().__init__(message.strip())
