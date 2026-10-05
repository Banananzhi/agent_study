import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemorySettings:
    # sqlite_path：长期记忆与 Outbox 使用的 SQLite 文件
    sqlite_path: str = ".agent_data/memories.sqlite3"
    # qdrant_url：Qdrant HTTP 服务地址
    qdrant_url: str = "http://localhost:6333"
    # qdrant_api_key：Qdrant Cloud 或受保护实例的访问密钥
    qdrant_api_key: str | None = None
    # qdrant_collection：长期记忆使用的 Collection 名称
    qdrant_collection: str = "agent_memories"
    # embedding_model：在线 Embedding 服务使用的模型名称
    embedding_model: str = "qwen3.7-text-embedding-flash"
    # embedding_base_url：Embedding 服务的 OpenAI 兼容 API 根地址
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # embedding_api_key：在线 Embedding 服务访问密钥
    embedding_api_key: str | None = field(default=None, repr=False)
    # embedding_timeout：单次向量请求超时秒数
    embedding_timeout: float = 30.0
    # embedding_dimensions：服务支持时可指定的目标向量维度
    embedding_dimensions: int | None = None
    # worker_batch_size：Worker 每次最多领取的任务数量
    worker_batch_size: int = 16
    # worker_lease_seconds：Worker 处理任务时持有的租约秒数
    worker_lease_seconds: int = 120
    # worker_max_attempts：索引任务允许执行的最大次数
    worker_max_attempts: int = 5
    # worker_poll_seconds：没有待处理任务时的轮询间隔秒数
    worker_poll_seconds: float = 2.0

    # 从环境变量创建长期记忆配置
    @classmethod
    def from_env(cls):
        # qdrant_api_key：空字符串需要转换为 None，避免发送空鉴权头
        qdrant_api_key = os.getenv("QDRANT_API_KEY") or None
        # embedding_api_key：优先使用 Agent 专用名称并兼容百炼标准变量
        embedding_api_key = (
            os.getenv("AGENT_EMBEDDING_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or None
        )
        # dimensions_value：服务未显式指定时让模型决定实际向量维度
        dimensions_value = os.getenv("AGENT_EMBEDDING_DIMENSIONS")
        return cls(
            sqlite_path=os.getenv("AGENT_MEMORY_DB_PATH", ".agent_data/memories.sqlite3"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=qdrant_api_key,
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "agent_memories"),
            embedding_model=os.getenv(
                "AGENT_EMBEDDING_MODEL",
                "qwen3.7-text-embedding-flash",
            ),
            embedding_base_url=os.getenv(
                "AGENT_EMBEDDING_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            embedding_api_key=embedding_api_key,
            embedding_timeout=float(os.getenv("AGENT_EMBEDDING_TIMEOUT", "30")),
            embedding_dimensions=(
                int(dimensions_value) if dimensions_value else None
            ),
            worker_batch_size=int(os.getenv("AGENT_MEMORY_WORKER_BATCH_SIZE", "16")),
            worker_lease_seconds=int(os.getenv("AGENT_MEMORY_WORKER_LEASE_SECONDS", "120")),
            worker_max_attempts=int(os.getenv("AGENT_MEMORY_WORKER_MAX_ATTEMPTS", "5")),
            worker_poll_seconds=float(os.getenv("AGENT_MEMORY_WORKER_POLL_SECONDS", "2")),
        )

