import ast
import copy
import ipaddress
import json
import operator
import os
import socket
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone as fixed_timezone
from enum import Enum
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel

from resource_lock import AccessMode, ResourceAccess
from tool_execution_policy import SideEffectLevel
from tool_models import (
    CalculatorOutput,
    CurrentTimeOutput,
    FileOperationOutput,
    ReadFileOutput,
    ReadWebpageOutput,
    WebSearchOutput,
)


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


# WORKSPACE_ROOT：允许文件工具读写的项目根目录
WORKSPACE_ROOT = Path(__file__).resolve().parent

# READ_FILE_CONTENT_JSON_BUDGET：在 8000 字符 Observation 中为文件内容预留的 JSON 长度
READ_FILE_CONTENT_JSON_BUDGET = 6000


class UnsafeRequestError(ValueError):
    pass


class ToolAuthenticationError(RuntimeError):
    pass


class ObservationPolicy(str, Enum):
    SUMMARIZE = "summarize"
    PAGINATE = "paginate"
    TRUNCATE = "truncate"
    RAW = "raw"


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
    return datetime.now(zone).replace(microsecond=0)


# 将模型提供的相对路径解析为工作区内的安全路径
# path：相对于项目根目录的文件路径
def _resolve_workspace_path(path):
    if not isinstance(path, str) or not path.strip():
        raise ValueError("文件路径不能为空")

    # requested_path：模型提供的原始相对路径
    requested_path = Path(path)
    if requested_path.is_absolute():
        raise UnsafeRequestError("不允许使用绝对路径")

    # workspace_root：解析过符号链接的工作区根目录
    workspace_root = WORKSPACE_ROOT.resolve()
    # target_path：解析过 .. 和符号链接的最终目标路径
    target_path = (workspace_root / requested_path).resolve(strict=False)
    try:
        # relative_path：用于返回给模型的规范化工作区相对路径
        relative_path = target_path.relative_to(workspace_root)
    except ValueError as error:
        raise UnsafeRequestError("文件路径不能越出工作区") from error
    if relative_path == Path("."):
        raise ValueError("文件路径不能指向工作区根目录")
    return target_path, relative_path.as_posix()


# 生成文件工具调用的规范化资源访问声明
# args：已通过工具输入 Schema 校验的参数字典
# mode：本次调用对目标文件的读写模式
def _file_resources(args, mode):
    # target_path：经过工作区边界和符号链接校验的目标路径
    target_path, _ = _resolve_workspace_path(args["path"])
    # resource_key：用于跨工具判断同一文件的大小写规范化绝对路径
    resource_key = os.path.normcase(os.path.normpath(str(target_path)))
    return (ResourceAccess("file", resource_key, mode),)


# 解析文件只读工具本次需要的资源
# args：已通过输入 Schema 校验的工具参数
def file_read_resources(args):
    return _file_resources(args, AccessMode.READ)


# 解析文件写入工具本次需要的独占资源
# args：已通过输入 Schema 校验的工具参数
def file_write_resources(args):
    return _file_resources(args, AccessMode.WRITE)


# 在工作区内创建 UTF-8 文件，已存在时拒绝覆盖
# path：相对于项目根目录的文件路径
# content：创建文件时写入的完整文本
def create_file(path, content=""):
    # target_path：通过工作区边界校验的文件路径
    target_path, relative_path = _resolve_workspace_path(path)
    if target_path.exists():
        raise ValueError(f"文件已存在：{relative_path}")

    # 自动创建工作区内缺失的父目录
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # chars_written：成功写入新文件的 Unicode 字符数
        with target_path.open("x", encoding="utf-8", newline="") as file:
            chars_written = file.write(content)
    except FileExistsError as error:
        raise ValueError(f"文件已存在：{relative_path}") from error
    return {
        "path": relative_path,
        "chars_written": chars_written,
        "next_offset": chars_written,
    }


# 按字符偏移覆盖首段或追加写入工作区内已存在的 UTF-8 文件
# path：相对于项目根目录的文件路径
# content：本次要写入的文本分段
# offset：本次写入的起始字符偏移，0 表示覆盖首段
def write_file(path, content, offset=0):
    # target_path：通过工作区边界校验的文件路径
    target_path, relative_path = _resolve_workspace_path(path)
    if not target_path.exists():
        raise ValueError(f"文件不存在：{relative_path}")
    if not target_path.is_file():
        raise ValueError(f"目标不是普通文件：{relative_path}")

    if offset == 0:
        # chars_written：成功覆盖写入首段的 Unicode 字符数
        chars_written = target_path.write_text(content, encoding="utf-8", newline="")
    else:
        # current_length：追加前文件已有的 Unicode 字符数
        with target_path.open("r", encoding="utf-8", newline="") as file:
            current_length = sum(len(chunk) for chunk in iter(lambda: file.read(8192), ""))
        if offset != current_length:
            raise ValueError(
                f"写入偏移不匹配：期望 {current_length}，实际传入 {offset}"
            )

        # chars_written：成功追加写入后续分段的 Unicode 字符数
        with target_path.open("a", encoding="utf-8", newline="") as file:
            chars_written = file.write(content)
    return {
        "path": relative_path,
        "chars_written": chars_written,
        "next_offset": offset + chars_written,
    }


# 读取工作区内已存在的 UTF-8 文件内容
# path：相对于项目根目录的文件路径
# offset：本次读取的起始字符偏移
# max_chars：每页最多返回的文件字符数
def read_file(path, offset=0, max_chars=6000):
    # target_path：通过工作区边界校验的文件路径
    target_path, relative_path = _resolve_workspace_path(path)
    if not target_path.exists():
        raise ValueError(f"文件不存在：{relative_path}")
    if not target_path.is_file():
        raise ValueError(f"目标不是普通文件：{relative_path}")

    # skipped_content：从文件开头跳过的已读取字符
    with target_path.open("r", encoding="utf-8", newline="") as file:
        skipped_content = file.read(offset)
        if len(skipped_content) != offset:
            raise ValueError(
                f"读取偏移超出文件长度：文件长度 {len(skipped_content)}，实际传入 {offset}"
            )

        # raw_content：多读取一个字符，用于判断是否还有下一页
        raw_content = file.read(max_chars + 1)

    # candidate_content：尚未考虑 JSON 转义膨胀的本页候选内容
    candidate_content = raw_content[:max_chars]
    # content_parts：在 Observation 内容预算内可安全返回的字符片段
    content_parts = []
    # serialized_chars：当前文件内容经 JSON 转义后占用的字符数
    serialized_chars = 0
    for character in candidate_content:
        # character_json_chars：当前字符经 JSON 转义后的实际长度
        character_json_chars = len(json.dumps(character, ensure_ascii=False)) - 2
        if serialized_chars + character_json_chars > READ_FILE_CONTENT_JSON_BUDGET:
            break
        content_parts.append(character)
        serialized_chars += character_json_chars

    # content：同时满足字符数和 Observation JSON 长度限制的本页内容
    content = "".join(content_parts)
    # has_more：文件在本页之后是否还有内容
    has_more = len(raw_content) > len(content)
    return {
        "path": relative_path,
        "offset": offset,
        "next_offset": offset + len(content),
        "has_more": has_more,
        "content": content,
    }


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
    output_model: type[BaseModel]
    retry_policy: RetryPolicy = RetryPolicy()
    idempotent: bool = True
    observation_policy: ObservationPolicy = ObservationPolicy.TRUNCATE
    resource_resolver: Callable[[dict], tuple] | None = None
    # side_effect_level：工具改变本地或外部状态的副作用等级
    side_effect_level: SideEffectLevel = SideEffectLevel.NONE

    # 校验工具的输入、输出 Schema 和显示名称
    def __post_init__(self):
        if not isinstance(self.schema, dict) or self.schema.get("type") != "function":
            raise ValueError("工具输入 Schema 必须是标准 function 类型")
        if not isinstance(self.output_model, type) or not issubclass(self.output_model, BaseModel):
            raise ValueError("工具 output_model 必须是 Pydantic BaseModel 子类")

        # output_schema：由 Pydantic 返回模型自动生成的 JSON Schema
        output_schema = self.output_model.model_json_schema()
        if not isinstance(output_schema.get("description"), str) or not output_schema["description"].strip():
            raise ValueError("工具 output_model 必须包含返回值说明")
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("工具 display_name 不能为空")
        if not isinstance(self.observation_policy, ObservationPolicy):
            raise ValueError("observation_policy 必须是 ObservationPolicy 枚举")
        if self.resource_resolver is not None and not callable(self.resource_resolver):
            raise ValueError("resource_resolver 必须为空或可调用对象")
        if not isinstance(self.side_effect_level, SideEffectLevel):
            raise ValueError("side_effect_level 必须是 SideEffectLevel 枚举")

    # 获取工具在注册表中的标准名称
    @property
    def name(self):
        return self.schema["function"]["name"]

    # 获取由 Pydantic 返回模型自动生成的 Output Schema
    @property
    def output_schema(self):
        return self.output_model.model_json_schema()

    # 根据已校验参数解析本次调用需要的全部资源
    # args：已通过输入 Schema 校验的工具参数
    def resolve_resources(self, args):
        if self.resource_resolver is None:
            return ()
        # resources：资源解析器为本次调用返回的资源声明
        resources = tuple(self.resource_resolver(args))
        if not all(isinstance(resource, ResourceAccess) for resource in resources):
            raise TypeError("resource_resolver 必须返回 ResourceAccess 序列")
        return resources


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

CREATE_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_file",
        "description": "在项目工作区内创建 UTF-8 文件；不会覆盖已存在的文件。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "相对于项目根目录的文件路径，例如 output/report.md。",
                    "minLength": 1,
                    "maxLength": 500,
                },
                "content": {
                    "type": "string",
                    "description": "新文件的完整文本内容，省略时创建空文件。",
                    "maxLength": 100000,
                    "default": "",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}

WRITE_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "按字符偏移写入已存在的 UTF-8 文件。offset=0 覆盖写入首段；"
            "后续分段必须使用上一次返回的 next_offset 追加。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "相对于项目根目录的已存在文件路径，例如 output/report.md。",
                    "minLength": 1,
                    "maxLength": 500,
                },
                "content": {
                    "type": "string",
                    "description": "本次要写入的文本分段。",
                    "maxLength": 100000,
                },
                "offset": {
                    "type": "integer",
                    "description": "本次写入的起始字符偏移；首段使用 0。",
                    "minimum": 0,
                    "default": 0,
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
}

READ_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "分页读取项目工作区内已存在的 UTF-8 文本文件。"
            "has_more 为 true 时，使用 next_offset 继续读取下一页。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "相对于项目根目录的已存在文件路径，例如 output/report.md。",
                    "minLength": 1,
                    "maxLength": 500,
                },
                "offset": {
                    "type": "integer",
                    "description": "本次读取的起始字符偏移；首页使用 0。",
                    "minimum": 0,
                    "default": 0,
                },
                "max_chars": {
                    "type": "integer",
                    "description": (
                        "每页最多返回的原始字符数；JSON 转义后过长时实际返回数可能更少。"
                    ),
                    "minimum": 1,
                    "maximum": 6000,
                    "default": 6000,
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}


TOOLS = {}


# 将工具函数及其完整 Schema 注册到工具表
# function：实际执行工具逻辑的函数
# schema：标准 Function Tool Schema
# output_model：用于校验、序列化和生成 Schema 的 Pydantic 返回模型
# display_name：ReAct 日志中的显示名称
# retry_policy：工具重试策略
# idempotent：工具能否安全地重复执行
# observation_policy：工具长结果进入模型上下文前的处理策略
# resource_resolver：根据已校验参数生成资源访问声明的函数
# side_effect_level：工具改变本地或外部状态的副作用等级
def register_tool(
    function,
    schema,
    output_model,
    display_name,
    retry_policy=None,
    idempotent=True,
    observation_policy=ObservationPolicy.TRUNCATE,
    resource_resolver=None,
    side_effect_level=SideEffectLevel.NONE,
):
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
        output_model=output_model,
        retry_policy=retry_policy or RetryPolicy(),
        idempotent=idempotent,
        observation_policy=observation_policy,
        resource_resolver=resource_resolver,
        side_effect_level=side_effect_level,
    )


register_tool(
    calculator,
    CALCULATOR_SCHEMA,
    CalculatorOutput,
    "Calculator",
    observation_policy=ObservationPolicy.RAW,
)
register_tool(
    web_search,
    WEB_SEARCH_SCHEMA,
    WebSearchOutput,
    "WebSearch",
    RetryPolicy(max_attempts=3),
    observation_policy=ObservationPolicy.SUMMARIZE,
)
register_tool(
    read_webpage,
    READ_WEBPAGE_SCHEMA,
    ReadWebpageOutput,
    "ReadWebpage",
    RetryPolicy(max_attempts=2),
    observation_policy=ObservationPolicy.SUMMARIZE,
)
register_tool(
    get_current_time,
    GET_CURRENT_TIME_SCHEMA,
    CurrentTimeOutput,
    "GetCurrentTime",
    observation_policy=ObservationPolicy.RAW,
)
register_tool(
    create_file,
    CREATE_FILE_SCHEMA,
    FileOperationOutput,
    "CreateFile",
    idempotent=False,
    observation_policy=ObservationPolicy.RAW,
    resource_resolver=file_write_resources,
    side_effect_level=SideEffectLevel.LOCAL_WRITE,
)
register_tool(
    write_file,
    WRITE_FILE_SCHEMA,
    FileOperationOutput,
    "WriteFile",
    idempotent=False,
    observation_policy=ObservationPolicy.RAW,
    resource_resolver=file_write_resources,
    side_effect_level=SideEffectLevel.LOCAL_WRITE,
)
register_tool(
    read_file,
    READ_FILE_SCHEMA,
    ReadFileOutput,
    "ReadFile",
    observation_policy=ObservationPolicy.PAGINATE,
    resource_resolver=file_read_resources,
)


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
    # schemas：将输出说明合并进 description 后的模型工具定义
    schemas = []

    # tool：当前正在生成模型定义的已注册工具
    for tool in active_registry.values():
        # schema：避免修改注册表原始数据的工具 Schema 副本
        schema = copy.deepcopy(tool.schema)

        # output_description：展示给模型的工具返回值语义
        output_description = tool.output_schema.get("description")
        if output_description:
            schema["function"]["description"] += f"\n返回值：{output_description}"
        schemas.append(schema)
    return schemas


# 将单个工具参数格式化为安全且有长度限制的日志文本
# name：工具参数名称
# value：模型为该参数生成的值
def _format_action_argument(name, value):
    # normalized_name：用于识别敏感字段和大文本字段的小写参数名
    normalized_name = name.lower()
    if any(marker in normalized_name for marker in ("password", "token", "secret", "api_key")):
        return "<已隐藏>"
    if normalized_name == "content" and isinstance(value, str):
        return f"<{len(value)} 字符>"

    # text：将复杂参数转换成稳定且不会转义中文的紧凑 JSON 文本
    if isinstance(value, str):
        text = value.replace("\r", "\\r").replace("\n", "\\n")
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    # max_chars：单个 Action 参数在日志中允许展示的最大字符数
    max_chars = 160
    return text if len(text) <= max_chars else text[:max_chars] + "…"


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

    if not isinstance(args, dict):
        return f"{label}[{_format_action_argument('arguments', args)}]"

    # schema_order：工具 Schema 中定义的稳定参数展示顺序
    schema_order = []
    if tool:
        schema_order = list(
            tool.schema["function"]["parameters"].get("properties", {})
        )
    # ordered_names：先按 Schema 排列，再保留不在 Schema 中的异常参数供排查
    ordered_names = [key for key in schema_order if key in args]
    ordered_names.extend(key for key in args if key not in ordered_names)
    # formatted_args：包含参数名称和值的可区分 Action 日志片段
    formatted_args = ", ".join(
        f"{key}={_format_action_argument(key, args[key])}"
        for key in ordered_names
    )
    return f"{label}[{formatted_args}]"


# 根据注册名称校验参数并执行工具
# name：工具注册名称
# args：以关键字参数形式传给工具函数的字典
def execute_tool(name, args):
    # tool：名称对应的已注册工具
    tool = TOOLS.get(name)
    if not tool:
        raise ValueError(f"未知工具：{name}")

    # 校验工具参数是否符合 Schema
    validate_tool_arguments(tool.schema, args)

    # value：工具函数返回的原始业务数据
    value = tool.function(**args)

    # validated_output：经过 Pydantic 返回模型校验的业务结果
    validated_output = tool.output_model.model_validate(value)
    return validated_output.model_dump(mode="json")
