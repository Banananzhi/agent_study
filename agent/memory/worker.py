import logging
import time

from agent.config import load_env
from agent.memory.config import MemorySettings
from agent.memory.factory import create_memory_service


logger = logging.getLogger(__name__)


# 持续处理 SQLite Outbox 中等待同步到 Qdrant 的任务
# service：已经组装完成的长期记忆服务
# poll_seconds：没有任务时的轮询间隔秒数
def run_worker(service, poll_seconds=2.0):
    if poll_seconds <= 0:
        raise ValueError("poll_seconds 必须大于 0")
    logger.info("长期记忆索引 Worker 已启动")
    while True:
        # report：本轮 Outbox 任务处理统计
        report = service.process_pending()
        if report.claimed:
            logger.info(
                "长期记忆索引完成：领取=%d，成功=%d，重试=%d，死信=%d，过期=%d",
                report.claimed,
                report.completed,
                report.retried,
                report.dead,
                report.superseded,
            )
            continue
        time.sleep(poll_seconds)


# 从 .env 创建服务并启动独立长期记忆索引 Worker
def main():
    load_env()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    # settings：环境变量生成的长期记忆配置
    settings = MemorySettings.from_env()
    # service：SQLite、Embedding 与 Qdrant 组成的长期记忆服务
    service = create_memory_service(settings)
    try:
        run_worker(service, settings.worker_poll_seconds)
    except KeyboardInterrupt:
        logger.info("长期记忆索引 Worker 已停止")


if __name__ == "__main__":
    main()

