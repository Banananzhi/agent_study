import logging
import os
import sys

from agent import Agent, AgentModelOutputError, AgentToolError, ContextProtocolError, ContextWindowError
from agent.memory import create_memory_service
from integrations.mcp import MCPClientManager, MCPServerConfig
from tooling.executor import ToolExecutor
from tooling.policy import SideEffectLevel
from tooling.registry import TOOLS


# 启动支持持久化上下文的多轮命令行对话
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # LangChain 底层 HTTP 客户端日志降级，避免混入非 Agent 流程的英文请求行
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    # memory_service：SQLite、Embedding、Qdrant 与 Outbox 组成的长期记忆服务
    memory_service = create_memory_service()
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
        # checkpoint_path：保存 LangGraph 会话状态的 SQLite 文件路径
        checkpoint_path = os.getenv(
            "AGENT_CHECKPOINT_PATH",
            ".agent_data/checkpoints.sqlite3",
        )
        # thread_id：本次命令行程序持续恢复的会话标识
        thread_id = os.getenv("AGENT_THREAD_ID", "default")
        with Agent(
            tool_executor=ToolExecutor(registry=registry),
            checkpoint_path=checkpoint_path,
            memory_service=memory_service,
        ) as agent:
            logging.getLogger(__name__).info("💬 当前会话: %s（输入 exit 退出）", thread_id)
            while True:
                question = input("User: ").strip()
                if question.lower() in {"exit", "quit", "退出"}:
                    break
                if not question:
                    print("请输入问题。")
                    continue
                try:
                    answer = agent.run(question, thread_id=thread_id)
                except (AgentToolError, AgentModelOutputError, ContextProtocolError, ContextWindowError) as error:
                    print(f"Assistant: 任务已终止，{error}")
                    continue
                print(f"Assistant: {answer}")
    finally:
        mcp_manager.close()
        memory_service.close()


if __name__ == "__main__":
    main()
