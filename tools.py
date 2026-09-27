import ast
import ipaddress
import json
import operator
import os
import socket
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone as fixed_timezone
from html.parser import HTMLParser
from typing import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


class UnsafeRequestError(ValueError):
    pass


class ToolAuthenticationError(RuntimeError):
    pass


# 安全计算基础算术表达式
# expression：仅包含数字及 + - * / ** 运算符的表达式
def calculator(expression):
    if not isinstance(expression, str) or len(expression) > 200:
        raise ValueError("算术表达式必须是长度不超过 200 的字符串")

    # 递归计算经过白名单限制的 AST 节点
    # node：当前待计算的 Python AST 节点
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("指数绝对值不能超过 100")
            return OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](visit(node.operand))
        raise ValueError("仅支持数字和 + - * / ** 运算")

    return visit(ast.parse(expression, mode="eval").body)


# 通过博查 API 搜索实时网页信息
# query：搜索关键词或自然语言问题
# count：希望返回的结果数量，范围限制在 1～10
def web_search(query, count=5):
    api_key = os.getenv("BOCHA_API_KEY")
    if not api_key or "请替换" in api_key:
        raise ToolAuthenticationError("请先在 .env 中设置 BOCHA_API_KEY")
    payload = json.dumps({
        "query": query,
        "freshness": "noLimit",
        "summary": True,
        "count": count,
    }).encode()
    request = urllib.request.Request(
        os.getenv("BOCHA_BASE_URL", "https://api.bochaai.com/v1/web-search"),
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.load(response)
    if result.get("code") not in (None, 200):
        raise RuntimeError(f"博查搜索失败：{result.get('msg', result)}")
    items = result.get("data", {}).get("webPages", {}).get("value", [])
    if not items:
        return "未搜索到相关结果"
    return "\n\n".join(
        f"[{index}] {item.get('name', '无标题')}\n{item.get('url', '')}\n"
        f"{item.get('summary') or item.get('snippet', '')}"
        for index, item in enumerate(items, 1)
    )


# 验证 URL 是否指向可公开访问的 HTTP/HTTPS 地址
# url：需要检查的网页地址
def _validate_public_url(url):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeRequestError("仅支持有效的 HTTP/HTTPS URL")
    default_port = 443 if parsed.scheme == "https" else 80
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or default_port, type=socket.SOCK_STREAM)
    for address in addresses:
        if not ipaddress.ip_address(address[4][0]).is_global:
            raise UnsafeRequestError("禁止访问本机或内网地址")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    # 跟随 HTTP 重定向前重新验证目标地址
    # request：原始 HTTP 请求对象
    # file：原始响应文件对象
    # code：HTTP 重定向状态码
    # message：HTTP 状态说明
    # headers：HTTP 响应头
    # new_url：重定向后的目标 URL
    def redirect_request(self, request, file, code, message, headers, new_url):
        _validate_public_url(new_url)
        return super().redirect_request(request, file, code, message, headers, new_url)


class _TextExtractor(HTMLParser):
    # 初始化 HTML 正文提取器
    def __init__(self):
        super().__init__()
        self.parts = []
        self.ignored_depth = 0

    # 处理 HTML 开始标签并标记需要忽略的内容区域
    # tag：HTML 标签名称
    # attrs：标签属性列表
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.ignored_depth += 1

    # 处理 HTML 结束标签并退出忽略区域
    # tag：HTML 标签名称
    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.ignored_depth:
            self.ignored_depth -= 1

    # 收集非脚本和非样式区域中的可见文本
    # data：HTML 解析器读取到的文本片段
    def handle_data(self, data):
        text = " ".join(data.split())
        if text and not self.ignored_depth:
            self.parts.append(text)


# 下载公开网页并提取可见正文
# url：需要读取的公开 HTTP/HTTPS 网页地址
# max_chars：返回正文的最大字符数
def read_webpage(url, max_chars=12000):
    _validate_public_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 AgentStudy/1.0"})
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    with opener.open(request, timeout=15) as response:
        content_type = response.headers.get_content_type()
        if content_type not in {"text/html", "text/plain"}:
            raise ValueError(f"不支持的网页类型：{content_type}")
        charset = response.headers.get_content_charset() or "utf-8"
        content = response.read(500_000).decode(charset, errors="replace")
    if content_type == "text/plain":
        return content[:max_chars]
    parser = _TextExtractor()
    parser.feed(content)
    return "\n".join(parser.parts)[:max_chars] or "网页没有可读取的正文"


# 获取指定时区的当前时间
# timezone：IANA 时区名称或 UTC±小时
def get_current_time(timezone="Asia/Shanghai"):
    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        common_offsets = {"Asia/Shanghai": 8, "Asia/Hong_Kong": 8, "UTC": 0}
        if timezone in common_offsets:
            zone = fixed_timezone(timedelta(hours=common_offsets[timezone]), name=timezone)
        elif timezone.startswith("UTC"):
            try:
                zone = fixed_timezone(timedelta(hours=float(timezone[3:] or 0)), name=timezone)
            except ValueError as error:
                raise ValueError(f"未知时区：{timezone}") from error
        else:
            raise ValueError(f"当前环境没有时区数据，暂不支持：{timezone}")
    return datetime.now(zone).isoformat(timespec="seconds")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    base_delay: float = 0.5
    max_delay: float = 2.0
    jitter: float = 0.2

    # 校验重试策略字段
    def __post_init__(self):
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts 必须是大于等于 1 的整数")
        if self.base_delay < 0 or self.max_delay < 0 or self.jitter < 0:
            raise ValueError("重试等待时间不能为负数")
        if self.base_delay > self.max_delay:
            raise ValueError("base_delay 不能大于 max_delay")


@dataclass(frozen=True)
class Tool:
    function: Callable[..., object]
    schema: dict
    display_name: str
    retry_policy: RetryPolicy = RetryPolicy()
    idempotent: bool = True

    # 获取工具在注册表中的标准名称
    @property
    def name(self):
        return self.schema["function"]["name"]


CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "计算基础算术表达式，适用于需要精确数学结果的任务。",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "仅包含数字和 + - * / ** 的算术表达式。",
                    "minLength": 1,
                    "maxLength": 200,
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}

WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "使用博查搜索实时网页信息，返回标题、URL 和摘要。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词或问题。", "minLength": 1},
                "count": {
                    "type": "integer",
                    "description": "返回结果数量。",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

READ_WEBPAGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_webpage",
        "description": "读取公开 HTTP/HTTPS 网页并提取可见正文。",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "需要读取的公开网页 URL。", "format": "uri"},
                "max_chars": {
                    "type": "integer",
                    "description": "返回正文的最大字符数。",
                    "minimum": 100,
                    "maximum": 50000,
                    "default": 12000,
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    },
}

GET_CURRENT_TIME_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "获取指定 IANA 时区或 UTC 偏移量的当前时间。",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "例如 Asia/Shanghai、UTC 或 UTC+8。",
                    "default": "Asia/Shanghai",
                }
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}


TOOLS = {}


# 将工具函数及其完整 Schema 注册到工具表
# function：实际执行工具逻辑的函数
# schema：标准 Function Tool Schema
# display_name：ReAct 日志中的显示名称
# retry_policy：工具重试策略
# idempotent：工具能否安全地重复执行
def register_tool(function, schema, display_name, retry_policy=None, idempotent=True):
    if schema.get("type") != "function" or not isinstance(schema.get("function"), dict):
        raise ValueError("工具 Schema 必须是标准 function 类型")
    definition = schema["function"]
    if not all(key in definition for key in ("name", "description", "parameters")):
        raise ValueError("工具 Schema 缺少 name、description 或 parameters")
    name = definition["name"]
    if name in TOOLS:
        raise ValueError(f"工具已注册：{name}")
    TOOLS[name] = Tool(
        function=function,
        schema=schema,
        display_name=display_name,
        retry_policy=retry_policy or RetryPolicy(),
        idempotent=idempotent,
    )


register_tool(calculator, CALCULATOR_SCHEMA, "Calculator")
register_tool(web_search, WEB_SEARCH_SCHEMA, "WebSearch", RetryPolicy(max_attempts=3))
register_tool(read_webpage, READ_WEBPAGE_SCHEMA, "ReadWebpage", RetryPolicy(max_attempts=2))
register_tool(get_current_time, GET_CURRENT_TIME_SCHEMA, "GetCurrentTime")


# 校验模型生成的工具参数是否符合 Schema
# schema：已注册工具的 Function Tool Schema
# args：模型生成的参数字典
def validate_tool_arguments(schema, args):
    parameters = schema["function"]["parameters"]
    if not isinstance(args, dict):
        raise ValueError("工具参数必须是 JSON 对象")
    missing = set(parameters.get("required", [])) - set(args)
    if missing:
        raise ValueError(f"缺少必填参数：{', '.join(sorted(missing))}")
    properties = parameters.get("properties", {})
    if parameters.get("additionalProperties") is False:
        extra = set(args) - set(properties)
        if extra:
            raise ValueError(f"存在未知参数：{', '.join(sorted(extra))}")
    type_map = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
    for name, value in args.items():
        rule = properties.get(name, {})
        expected = type_map.get(rule.get("type"))
        if expected and (not isinstance(value, expected) or isinstance(value, bool) and rule["type"] != "boolean"):
            raise ValueError(f"参数 {name} 类型错误，应为 {rule['type']}")
        if isinstance(value, str) and len(value) < rule.get("minLength", 0):
            raise ValueError(f"参数 {name} 长度不足")
        if isinstance(value, str) and len(value) > rule.get("maxLength", float("inf")):
            raise ValueError(f"参数 {name} 长度超限")
        if isinstance(value, (int, float)) and value < rule.get("minimum", -float("inf")):
            raise ValueError(f"参数 {name} 小于最小值")
        if isinstance(value, (int, float)) and value > rule.get("maximum", float("inf")):
            raise ValueError(f"参数 {name} 大于最大值")


# 获取模型原生 Function Calling 所需的工具 Schema
# registry：当前 Agent 实际使用的工具注册表
def get_tool_schemas(registry=None):
    # active_registry：用于生成 Schema 的实际工具注册表
    active_registry = TOOLS if registry is None else registry
    return [tool.schema for tool in active_registry.values()]


# 将工具调用格式化为 ReAct 日志中的 Action 文本
# name：工具注册名称
# args：工具参数字典
# registry：当前 Agent 实际使用的工具注册表
def format_tool_action(name, args, registry=None):
    # active_registry：用于查找日志显示名称的工具注册表
    active_registry = TOOLS if registry is None else registry

    # tool：工具名称对应的注册信息
    tool = active_registry.get(name)

    # label：日志中使用的工具显示名称
    label = tool.display_name if tool else name

    # value：日志中展示的首个工具参数值
    value = next(iter(args.values()), "") if isinstance(args, dict) else args
    return f"{label}[{value}]"


# 根据注册名称校验参数并执行工具
# name：工具注册名称
# args：以关键字参数形式传给工具函数的字典
def execute_tool(name, args):
    tool = TOOLS.get(name)
    if not tool:
        raise ValueError(f"未知工具：{name}")
    # 校验工具参数是否符合 Schema
    validate_tool_arguments(tool.schema, args)
    return tool.function(**args)
