from agent.memory.config import MemorySettings
from agent.memory.embeddings import OpenAICompatibleEmbeddingProvider
from agent.memory.qdrant_index import QdrantMemoryIndex
from agent.memory.repository import MemoryRepository
from agent.memory.service import LongTermMemoryService, MemoryRetryPolicy


# 使用环境配置组装长期记忆存储、索引与可靠双写服务
# settings：可选的显式长期记忆配置
def create_memory_service(settings=None):
    # resolved_settings：显式配置或环境变量生成的最终配置
    resolved_settings = settings or MemorySettings.from_env()
    # repository：SQLite 长期记忆与 Outbox Repository
    repository = MemoryRepository(resolved_settings.sqlite_path)
    # embedder：通过百炼 OpenAI 兼容接口调用的在线 Embedding 组件
    embedder = OpenAICompatibleEmbeddingProvider(
        model_name=resolved_settings.embedding_model,
        base_url=resolved_settings.embedding_base_url,
        api_key=resolved_settings.embedding_api_key,
        timeout=resolved_settings.embedding_timeout,
        dimensions=resolved_settings.embedding_dimensions,
    )
    # index：只保存向量和过滤字段的 Qdrant 索引客户端
    index = QdrantMemoryIndex(
        url=resolved_settings.qdrant_url,
        api_key=resolved_settings.qdrant_api_key,
        collection_name=resolved_settings.qdrant_collection,
    )
    # retry_policy：Outbox 任务失败后的最大自动执行次数
    retry_policy = MemoryRetryPolicy(
        max_attempts=resolved_settings.worker_max_attempts,
    )
    return LongTermMemoryService(
        repository=repository,
        embedder=embedder,
        index=index,
        retry_policy=retry_policy,
        batch_size=resolved_settings.worker_batch_size,
        lease_seconds=resolved_settings.worker_lease_seconds,
    )

