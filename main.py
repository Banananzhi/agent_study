import logging
import os
import sys

from agent import Agent, AgentToolError
from mcp_client import MCPClientManager, MCPServerConfig
from tool_execution_policy import SideEffectLevel
from tool_executor import ToolExecutor
from tools import TOOLS


# 启动一次单轮命令行对话
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # LangChain 底层 HTTP 客户端日志降级，避免混入非 Agent 流程的英文请求行
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    # mcp_manager：维护远程 DeepWiki MCP Session 的持久 Client 管理器
    mcp_manager = MCPClientManager([
        MCPServerConfig(
            name="deepwiki",
            url=os.getenv("DEEPWIKI_MCP_URL", "https://mcp.deepwiki.com/mcp"),
            timeout=30,
            default_side_effect_level=SideEffectLevel.NONE,
        )
    ])
    try:
        try:
            # mcp_tools：启动时通过 list_tools 动态发现的远程工具注册表
            mcp_tools = mcp_manager.connect()
        except Exception as error:
            logger = logging.getLogger(__name__)
            logger.warning("⚠️ MCP Server 连接失败，仅启用本地工具：%s", error)
            mcp_tools = {}
        # FastMCP 初始化可能调整第三方日志级别，合并工具前再次恢复静默配置
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpx2").setLevel(logging.WARNING)
        # registry：当前 Agent 独享的本地与 MCP 工具合并注册表
        registry = {**TOOLS, **mcp_tools}
        agent = Agent(tool_executor=ToolExecutor(registry=registry))
        question = input("User: ").strip()
        if not question:
            print("请输入问题。")
            return
        try:
            answer = agent.run(question)
        except AgentToolError as error:
            print(f"Assistant: 任务已终止，{error}")
            return
        print(f"Assistant: {answer}")
    finally:
        mcp_manager.close()


if __name__ == "__main__":
    main()
