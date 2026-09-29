import asyncio
import logging
import re
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any

from fastmcp import Client
from pydantic import RootModel

from tool_execution_policy import SideEffectLevel
from tools import ObservationPolicy, Tool


logger = logging.getLogger(__name__)


class MCPToolOutput(RootModel[Any]):
    """返回 MCP Server 提供的结构化数据或文本内容。"""


class MCPToolExecutionError(ValueError):
    pass


@dataclass(frozen=True)
class MCPServerConfig:
    # name：用于工具命名空间和连接查找的本地 Server 名称
    name: str
    # url：远程 MCP Server 的 Streamable HTTP 地址
    url: str
    # timeout：连接、发现和单次工具调用的超时秒数
    timeout: float = 30
    # default_side_effect_level：Server 未声明 annotations 时采用的副作用等级
    default_side_effect_level: SideEffectLevel = SideEffectLevel.EXTERNAL_WRITE

    # 校验远程 MCP Server 配置
    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("MCP Server name 不能为空")
        if not isinstance(self.url, str) or not self.url.startswith(("http://", "https://")):
            raise ValueError("MCP Server url 必须是 HTTP/HTTPS 地址")
        if not isinstance(self.timeout, (int, float)) or self.timeout <= 0:
            raise ValueError("MCP Server timeout 必须是正数")
        if not isinstance(self.default_side_effect_level, SideEffectLevel):
            raise TypeError("default_side_effect_level 必须是 SideEffectLevel")


# 将名称转换为模型 Function Calling 可安全使用的标识符
# value：Server 或远程工具的原始名称
def normalize_tool_name(value):
    # normalized：将非字母、数字和下划线字符统一替换后的名称
    normalized = re.sub(r"[^a-zA-Z0-9_]", "_", value.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise ValueError("MCP 名称规范化后不能为空")
    return normalized


class MCPToolAdapter:
    # 初始化将远程 MCP Tool 转换为项目 Tool 的适配器
    # manager：负责执行远程 call_tool 的 MCP Client 管理器
    # config：当前远程 Server 的连接和默认安全配置
    def __init__(self, manager, config):
        self.manager = manager
        self.config = config

    # 从 MCP annotations 推导项目副作用等级
    # remote_tool：FastMCP list_tools 返回的远程工具定义
    def _side_effect_level(self, remote_tool):
        # annotations：MCP Server 对工具行为提供的提示信息
        annotations = getattr(remote_tool, "annotations", None)
        if annotations is not None:
            if getattr(annotations, "destructive_hint", False):
                return SideEffectLevel.DESTRUCTIVE
            if getattr(annotations, "read_only_hint", False):
                return SideEffectLevel.NONE
        return self.config.default_side_effect_level

    # 从 MCP annotations 推导工具是否可以安全重试
    # remote_tool：FastMCP list_tools 返回的远程工具定义
    def _idempotent(self, remote_tool):
        # annotations：可能包含 idempotent_hint 的 MCP 工具行为提示
        annotations = getattr(remote_tool, "annotations", None)
        if annotations is not None:
            idempotent_hint = getattr(annotations, "idempotent_hint", None)
            if isinstance(idempotent_hint, bool):
                return idempotent_hint
        return self._side_effect_level(remote_tool) == SideEffectLevel.NONE

    # 将一个远程 MCP Tool 适配为可加入当前注册表的项目 Tool
    # remote_tool：FastMCP list_tools 返回的远程工具定义
    def adapt(self, remote_tool):
        # remote_name：调用 MCP Server 时必须使用的原始工具名称
        remote_name = remote_tool.name
        # registered_name：加入模型工具列表的带 Server 命名空间名称
        registered_name = (
            f"{normalize_tool_name(self.config.name)}__"
            f"{normalize_tool_name(remote_name)}"
        )
        # input_schema：MCP SDK v2 提供的标准工具输入 JSON Schema
        input_schema = dict(remote_tool.input_schema or {"type": "object"})
        input_schema.setdefault("type", "object")
        input_schema.setdefault("properties", {})
        input_schema.setdefault("additionalProperties", False)
        # description：告知模型工具来源和远程 Server 提供的用途说明
        description = remote_tool.description or f"调用 {self.config.name} MCP 工具"

        # invoke：ToolExecutor 在线程池内调用的同步 MCP 工具代理函数
        # arguments：模型经过 Schema 校验后的远程工具参数
        def invoke(**arguments):
            return self.manager.call_tool(
                self.config.name,
                remote_name,
                arguments,
            )

        return Tool(
            function=invoke,
            schema={
                "type": "function",
                "function": {
                    "name": registered_name,
                    "description": f"[MCP:{self.config.name}] {description}",
                    "parameters": input_schema,
                },
            },
            display_name=f"MCP:{self.config.name}/{remote_name}",
            output_model=MCPToolOutput,
            idempotent=self._idempotent(remote_tool),
            observation_policy=ObservationPolicy.SUMMARIZE,
            side_effect_level=self._side_effect_level(remote_tool),
        )


class MCPClientManager:
    # 初始化在独立 asyncio 事件循环中维护持久连接的 FastMCP Client 管理器
    # configs：需要连接的远程 MCP Server 配置列表
    def __init__(self, configs):
        # config_by_name：按唯一 Server 名称索引的配置表
        config_by_name = {config.name: config for config in configs}
        if len(config_by_name) != len(configs):
            raise ValueError("MCP Server name 不能重复")
        self.configs = config_by_name
        self.clients = {}
        self.remote_tools = {}
        self.loop = asyncio.new_event_loop()
        self.loop_ready = threading.Event()
        self.thread = threading.Thread(
            target=self._run_event_loop,
            name="mcp-client-event-loop",
            daemon=True,
        )
        self.thread.start()
        if not self.loop_ready.wait(5):
            raise RuntimeError("MCP Client 事件循环启动超时")
        self.closed = False

    # 在后台线程中启动并持续运行 MCP asyncio 事件循环
    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop_ready.set()
        self.loop.run_forever()

    # 将协程安全提交到 MCP 事件循环并同步等待结果
    # coroutine：必须在 MCP 后台事件循环中执行的协程对象
    # timeout：同步调用方允许等待的最长秒数
    def _submit(self, coroutine, timeout):
        if self.closed:
            coroutine.close()
            raise RuntimeError("MCPClientManager 已关闭")
        # future：跨线程提交到持久事件循环的协程结果
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as error:
            future.cancel()
            raise TimeoutError("MCP 操作超时") from error

    # 连接全部 Server、发现工具并返回可与本地工具合并的注册表
    def connect(self):
        # total_timeout：为多 Server 首次连接预留的总等待时间
        total_timeout = sum(config.timeout for config in self.configs.values()) + 5
        return self._submit(self._connect_all(), total_timeout)

    # 在后台事件循环中建立全部持久 MCP 连接
    async def _connect_all(self):
        # registry：本次发现并完成命名空间处理的 MCP Tool 注册表
        registry = {}
        try:
            # config：当前正在连接和发现工具的远程 Server 配置
            for config in self.configs.values():
                logger.info("🔌 正在连接 MCP Server: %s", config.name)
                # client：当前 Server 对应的持久 FastMCP Client
                client = Client(
                    config.url,
                    name=f"agent-study/{config.name}",
                    timeout=config.timeout,
                )
                await client.__aenter__()
                self.clients[config.name] = client
                # tools：通过标准 list_tools 发现的远程工具定义
                tools = await client.list_tools()
                self.remote_tools[config.name] = tools
                # adapter：将当前 Server 工具转换为项目 Tool 的适配器
                adapter = MCPToolAdapter(self, config)
                for remote_tool in tools:
                    # tool：已经带 Server 命名空间的项目工具定义
                    tool = adapter.adapt(remote_tool)
                    if tool.name in registry:
                        raise ValueError(f"MCP 工具名称冲突：{tool.name}")
                    registry[tool.name] = tool
                logger.info("✅ MCP Server %s 已发现 %d 个工具", config.name, len(tools))
            return registry
        except Exception:
            await self._close_all()
            raise

    # 同步调用一个已经连接的远程 MCP 工具
    # server_name：目标 Server 的本地配置名称
    # tool_name：MCP Server 公布的原始工具名称
    # arguments：已经通过工具 Schema 校验的参数字典
    def call_tool(self, server_name, tool_name, arguments):
        config = self.configs.get(server_name)
        if config is None:
            raise MCPToolExecutionError(f"未知 MCP Server：{server_name}")
        return self._submit(
            self._call_tool(server_name, tool_name, arguments),
            config.timeout + 1,
        )

    # 在 MCP 事件循环中执行 call_tool 并规范化返回数据
    # server_name：目标 Server 的本地配置名称
    # tool_name：MCP Server 公布的原始工具名称
    # arguments：远程工具调用参数
    async def _call_tool(self, server_name, tool_name, arguments):
        client = self.clients.get(server_name)
        if client is None:
            raise MCPToolExecutionError(f"MCP Server 尚未连接：{server_name}")
        logger.info("🌐 正在调用 MCP 工具: %s/%s", server_name, tool_name)
        # result：FastMCP 对标准 CallToolResult 的便捷封装
        result = await client.call_tool(
            tool_name,
            arguments,
            raise_on_error=False,
        )
        if result.is_error:
            # message：合并远程文本错误块后返回统一工具错误
            message = "\n".join(
                getattr(item, "text", str(item))
                for item in result.content
            ) or "MCP Server 返回工具执行错误"
            raise MCPToolExecutionError(message)
        if result.data is not None:
            return result.data
        if result.structured_content is not None:
            return result.structured_content
        return "\n".join(
            getattr(item, "text", str(item))
            for item in result.content
        )

    # 关闭所有 MCP Session 和后台事件循环
    def close(self):
        if self.closed:
            return
        try:
            self._submit(self._close_all(), 10)
        finally:
            self.closed = True
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)
            self.loop.close()

    # 在后台事件循环中退出所有 FastMCP Client 上下文
    async def _close_all(self):
        # client：当前需要关闭的持久 FastMCP Client
        for client in reversed(list(self.clients.values())):
            await client.__aexit__(None, None, None)
        self.clients.clear()
        self.remote_tools.clear()
