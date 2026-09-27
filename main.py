import logging
import sys

from agent import Agent, AgentToolError


# 启动一次单轮命令行对话
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    agent = Agent()
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


if __name__ == "__main__":
    main()
