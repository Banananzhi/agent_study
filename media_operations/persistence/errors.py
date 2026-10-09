class NotFoundError(LookupError):
    """对象不存在或不属于当前可信身份/账号。"""


class ConflictError(ValueError):
    """版本、幂等键或状态冲突。"""
