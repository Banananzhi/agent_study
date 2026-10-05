import logging
from dataclasses import dataclass

import httpx
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from agent.memory.models import (
    IndexJobStatus,
    IndexOperation,
    IndexRunReport,
    MemoryStatus,
    MemoryWrite,
)
from agent.memory.qdrant_index import MemoryIndexConfigurationError


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryRetryPolicy:
    # max_attempts：单个索引任务允许执行的最大次数
    max_attempts: int = 5
    # base_delay_seconds：指数退避的初始等待秒数
    base_delay_seconds: int = 5
    # max_delay_seconds：指数退避允许的最大等待秒数
    max_delay_seconds: int = 3600

    # 根据当前执行次数计算下一次重试等待时间
    # attempts：任务已经执行的次数
    def delay_for(self, attempts):
        return min(self.base_delay_seconds * (2 ** max(attempts - 1, 0)), self.max_delay_seconds)


class LongTermMemoryService:
    # 初始化 SQLite 主存储、Embedding 和 Qdrant 索引的协调服务
    # repository：长期记忆 SQLite Repository
    # embedder：文本向量生成组件
    # index：Qdrant 长期记忆索引
    # retry_policy：Outbox 失败重试策略
    # batch_size：每次最多处理的 Outbox 任务数
    # lease_seconds：任务处理租约秒数
    def __init__(
        self,
        repository,
        embedder,
        index,
        retry_policy=None,
        batch_size=16,
        lease_seconds=120,
    ):
        self.repository = repository
        self.embedder = embedder
        self.index = index
        self.retry_policy = retry_policy or MemoryRetryPolicy()
        self.batch_size = batch_size
        self.lease_seconds = lease_seconds

    # 释放长期记忆索引持有的外部连接
    def close(self):
        # embedder_close：在线 Embedding Provider 可选提供的连接释放方法
        embedder_close = getattr(self.embedder, "close", None)
        if callable(embedder_close):
            embedder_close()
        # close：Qdrant 索引可选提供的连接释放方法
        close = getattr(self.index, "close", None)
        if callable(close):
            close()

    # 为当前用户轮次领取一次幂等记忆提取执行权
    # turn_id：当前用户轮次的稳定唯一标识
    # thread_id：当前会话标识
    # user_message_id：触发当前轮次的用户消息标识
    def begin_extraction(self, turn_id, thread_id, user_message_id):
        return self.repository.begin_extraction(
            turn_id,
            thread_id,
            user_message_id,
        )

    # 将当前用户轮次标记为已经完成记忆提取
    # turn_id：当前用户轮次的稳定唯一标识
    # extracted_count：本轮实际写入的长期记忆数量
    def complete_extraction(self, turn_id, extracted_count):
        self.repository.complete_extraction(turn_id, extracted_count)

    # 记录当前用户轮次的记忆提取失败
    # turn_id：当前用户轮次的稳定唯一标识
    # error：提取模型或存储层返回的异常
    def fail_extraction(self, turn_id, error):
        self.repository.fail_extraction(turn_id, error)

    # 将记忆和 Outbox 任务原子写入 SQLite
    # value：经过校验的长期记忆写入请求
    def remember(self, value):
        if not isinstance(value, MemoryWrite):
            value = MemoryWrite.model_validate(value)
        return self.repository.upsert(value, self.embedder.model_name)

    # 软删除 SQLite 记忆并可靠创建 Qdrant 删除任务
    # memory_id：需要删除的记忆标识
    # tenant_id：经过身份认证的租户标识
    # user_id：经过身份认证的用户标识
    def forget(self, memory_id, tenant_id, user_id):
        return self.repository.soft_delete(memory_id, tenant_id, user_id)

    # 判断索引异常是否适合由 Outbox 自动重试
    # error：Embedding 或 Qdrant 抛出的异常
    @staticmethod
    def _is_retryable(error):
        if isinstance(error, httpx.HTTPStatusError):
            # status_code：Embedding 服务返回的 HTTP 状态码
            status_code = error.response.status_code
            return status_code == 429 or status_code >= 500
        if isinstance(error, (httpx.TimeoutException, httpx.NetworkError)):
            return True
        if isinstance(error, UnexpectedResponse):
            return error.status_code == 429 or error.status_code >= 500
        if isinstance(error, (ResponseHandlingException, TimeoutError, ConnectionError, OSError)):
            return True
        if isinstance(error, (ValueError, TypeError, MemoryIndexConfigurationError)):
            return False
        return False

    # 执行一条已经领取的 Qdrant 索引任务
    # job：处于 processing 状态的 Outbox 任务
    def _process_job(self, job):
        # memory：任务执行时从 SQLite 重新读取的最新事实
        memory = self.repository.get(job.memory_id)
        if memory is None or memory.version != job.memory_version:
            self.repository.supersede_job(job.event_id)
            return IndexJobStatus.SUPERSEDED

        if job.operation is IndexOperation.DELETE or memory.status is MemoryStatus.DELETED:
            self.index.delete(memory.id)
        else:
            # vectors：长期记忆正文生成的单条文档向量
            vectors = self.embedder.embed_documents([memory.content])
            if len(vectors) != 1 or not vectors[0]:
                raise ValueError("EmbeddingProvider 必须返回一条非空向量")
            self.index.upsert(memory, vectors[0])

        # is_current：同步完成时任务是否仍对应 SQLite 最新版本
        is_current = self.repository.complete_job(job.event_id)
        return IndexJobStatus.COMPLETED if is_current else IndexJobStatus.SUPERSEDED

    # 处理一批 Outbox 任务，不在失败任务上阻塞等待
    def process_pending(self):
        # jobs：本次通过 SQLite 租约成功领取的任务
        jobs = self.repository.claim_jobs(self.batch_size, self.lease_seconds)
        # report：本批索引处理统计
        report = IndexRunReport(claimed=len(jobs))
        for job in jobs:
            try:
                # status：单条任务最终处理状态
                status = self._process_job(job)
                if status is IndexJobStatus.COMPLETED:
                    report.completed += 1
                else:
                    report.superseded += 1
            except Exception as error:
                # retryable：当前异常是否允许自动重试
                retryable = self._is_retryable(error)
                # retry_delay：任务重新进入可领取状态前的等待时间
                retry_delay = self.retry_policy.delay_for(job.attempts)
                # failure_status：Repository 根据次数和错误策略决定的状态
                failure_status = self.repository.fail_job(
                    job.event_id,
                    error,
                    retryable,
                    self.retry_policy.max_attempts,
                    retry_delay,
                )
                if failure_status is IndexJobStatus.PENDING:
                    report.retried += 1
                elif failure_status is IndexJobStatus.SUPERSEDED:
                    report.superseded += 1
                else:
                    report.dead += 1
                logger.warning(
                    "长期记忆索引任务失败：memory_id=%s，attempt=%d，retryable=%s，error=%s",
                    job.memory_id,
                    job.attempts,
                    retryable,
                    error,
                )
        return report

    # 在可信用户边界内检索相关记忆并从 SQLite 批量加载正文
    # query：当前任务或问题文本
    # tenant_id：经过身份认证的租户标识
    # user_id：经过身份认证的用户标识
    # project_id：需要限定的项目标识
    # memory_types：允许召回的记忆类型
    # limit：最多召回的记忆数量
    # score_threshold：最低语义相关度
    def recall(
        self,
        query,
        tenant_id,
        user_id,
        project_id=None,
        memory_types=None,
        limit=8,
        score_threshold=None,
    ):
        # 没有已索引记忆时跳过模型加载与 Qdrant 请求
        if not self.repository.has_indexed_memories(tenant_id, user_id):
            return []
        # query_vector：当前查询生成的向量
        query_vector = self.embedder.embed_query(query)
        # hits：经过 Qdrant 租户和用户条件过滤的相关记忆标识
        hits = self.index.search(
            query_vector,
            tenant_id,
            user_id,
            project_id=project_id,
            memory_types=memory_types,
            limit=limit,
            score_threshold=score_threshold,
        )
        # memories：按 Qdrant 相关度顺序批量加载的 SQLite 完整记忆
        memories = self.repository.get_many(
            [hit.memory_id for hit in hits],
            tenant_id,
            user_id,
        )
        # scores：用于将相关度与完整记忆重新组合的分数映射
        scores = {hit.memory_id: hit.score for hit in hits}
        return [(memory, scores[memory.id]) for memory in memories]

