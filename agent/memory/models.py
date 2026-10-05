from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoryType(str, Enum):
    PREFERENCE = "preference"
    FACT = "fact"
    DECISION = "decision"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    DOCUMENT_SUMMARY = "document_summary"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    DELETED = "deleted"
    SUPERSEDED = "superseded"


class IndexStatus(str, Enum):
    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"


class IndexOperation(str, Enum):
    UPSERT = "upsert"
    DELETE = "delete"


class IndexJobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    DEAD = "dead"
    SUPERSEDED = "superseded"


class MemoryWrite(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    # tenant_id：经过身份认证的租户标识
    tenant_id: str = Field(min_length=1, max_length=128)
    # user_id：经过身份认证的用户标识
    user_id: str = Field(min_length=1, max_length=128)
    # project_id：记忆所属项目，不属于具体项目时为空
    project_id: str | None = Field(default=None, max_length=128)
    # memory_type：长期记忆的业务类型
    memory_type: MemoryType
    # memory_key：支持精确更新和去重的稳定键，经历型记忆可以为空
    memory_key: str | None = Field(default=None, max_length=256)
    # content：长期记忆的完整文本
    content: str = Field(min_length=1)
    # importance：记忆对后续任务的重要程度
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    # confidence：记忆内容可信程度
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    # source_thread_id：产生该记忆的会话标识
    source_thread_id: str | None = Field(default=None, max_length=256)
    # source_message_ids：支持回溯证据的原始消息标识
    source_message_ids: list[str] = Field(default_factory=list)
    # expires_at：记忆过期时间，None 表示长期有效
    expires_at: datetime | None = None

    # 校验来源消息标识不包含空值并保持顺序去重
    # value：待校验的来源消息标识列表
    @field_validator("source_message_ids")
    @classmethod
    def validate_source_message_ids(cls, value):
        # normalized：保持原始顺序的消息标识集合
        normalized = []
        # seen：已经加入结果的消息标识
        seen = set()
        for message_id in value:
            if not isinstance(message_id, str) or not message_id.strip():
                raise ValueError("source_message_ids 不能包含空值")
            # clean_message_id：去除首尾空白后的消息标识
            clean_message_id = message_id.strip()
            if clean_message_id not in seen:
                seen.add(clean_message_id)
                normalized.append(clean_message_id)
        return normalized


class LongTermMemory(MemoryWrite):
    # id：SQLite 与 Qdrant 共享的记忆唯一标识
    id: str
    # version：用于阻止旧索引任务覆盖新内容的单调版本号
    version: int = Field(ge=1)
    # content_hash：用于快速判断文本是否发生变化的摘要
    content_hash: str
    # status：记忆当前生命周期状态
    status: MemoryStatus
    # index_status：该版本在 Qdrant 中的索引状态
    index_status: IndexStatus
    # embedding_model：当前索引使用的向量模型名称
    embedding_model: str
    # created_at：记忆首次创建时间
    created_at: datetime
    # updated_at：记忆最近更新时间
    updated_at: datetime


class MemoryIndexJob(BaseModel):
    # event_id：Outbox 任务唯一标识
    event_id: str
    # memory_id：需要建立或删除索引的记忆标识
    memory_id: str
    # memory_version：任务创建时对应的记忆版本
    memory_version: int = Field(ge=1)
    # operation：本次需要执行的索引操作
    operation: IndexOperation
    # status：Outbox 任务处理状态
    status: IndexJobStatus
    # attempts：任务已经被 Worker 领取的次数
    attempts: int = Field(ge=0)
    # next_retry_at：失败任务允许再次处理的时间
    next_retry_at: datetime | None
    # locked_until：Worker 租约到期时间
    locked_until: datetime | None
    # last_error：最近一次执行失败的安全错误信息
    last_error: str | None
    # created_at：任务创建时间
    created_at: datetime
    # updated_at：任务最近更新时间
    updated_at: datetime


class MemorySearchHit(BaseModel):
    # memory_id：语义检索命中的记忆标识
    memory_id: str
    # score：Qdrant 返回的语义相关度分数
    score: float


class IndexRunReport(BaseModel):
    # claimed：本次领取的 Outbox 任务数量
    claimed: int = 0
    # completed：成功同步到 Qdrant 的任务数量
    completed: int = 0
    # retried：等待后续重试的任务数量
    retried: int = 0
    # dead：不可恢复或重试耗尽的任务数量
    dead: int = 0
    # superseded：因为版本落后而跳过的任务数量
    superseded: int = 0


class MemoryCandidate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    # memory_type：候选记忆的业务类型
    memory_type: MemoryType
    # memory_key：偏好、事实和决策使用的稳定语义键
    memory_key: str | None = Field(default=None, max_length=256)
    # content：脱离当前对话后仍可理解的完整记忆文本
    content: str = Field(min_length=1, max_length=4000)
    # importance：候选记忆对后续任务的重要程度
    importance: float = Field(ge=0.0, le=1.0)
    # confidence：候选内容由当前轮次明确支持的程度
    confidence: float = Field(ge=0.0, le=1.0)
    # expires_at：具有明显时效性的记忆过期时间
    expires_at: datetime | None = None


class MemoryExtractionBatch(BaseModel):
    # candidates：当前用户轮次提取的有限候选记忆
    candidates: list[MemoryCandidate] = Field(default_factory=list, max_length=5)

