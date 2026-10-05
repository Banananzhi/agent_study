import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent.memory.models import (
    IndexJobStatus,
    IndexOperation,
    IndexStatus,
    LongTermMemory,
    MemoryIndexJob,
    MemoryStatus,
    MemoryWrite,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    project_id TEXT,
    project_scope TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    memory_key TEXT,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    importance REAL NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,
    index_status TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    version INTEGER NOT NULL,
    source_thread_id TEXT,
    source_message_ids TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_active_memory_key
ON memories(tenant_id, user_id, project_scope, memory_type, memory_key)
WHERE memory_key IS NOT NULL AND status = 'active';

CREATE INDEX IF NOT EXISTS idx_memory_scope
ON memories(tenant_id, user_id, project_scope, status);

CREATE INDEX IF NOT EXISTS idx_memory_index_status
ON memories(index_status, status);

CREATE TABLE IF NOT EXISTS memory_index_jobs (
    event_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    memory_version INTEGER NOT NULL,
    operation TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_retry_at TEXT,
    locked_until TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(memory_id, memory_version, operation),
    FOREIGN KEY(memory_id) REFERENCES memories(id)
);

CREATE INDEX IF NOT EXISTS idx_memory_jobs_ready
ON memory_index_jobs(status, next_retry_at, created_at);

CREATE INDEX IF NOT EXISTS idx_memory_jobs_memory
ON memory_index_jobs(memory_id, status, memory_version);

CREATE TABLE IF NOT EXISTS memory_extraction_runs (
    turn_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    user_message_id TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    extracted_count INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT,
    last_error TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_extraction_status
ON memory_extraction_runs(status, locked_until, updated_at);
"""


class MemoryRepository:
    # 初始化长期记忆 SQLite Repository 并创建所需表结构
    # path：SQLite 数据库文件路径
    def __init__(self, path=".agent_data/memories.sqlite3"):
        self.path = str(path)
        # database_path：用于确保父目录存在的数据库路径
        database_path = Path(self.path)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    # 创建带统一安全配置的 SQLite 连接
    def _open_connection(self):
        # connection：当前 Repository 操作使用的独立数据库连接
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    # 管理一次 SQLite 操作并确保连接最终关闭
    @contextmanager
    def _connect(self):
        # connection：本次操作独占的 SQLite 连接
        connection = self._open_connection()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    # 初始化数据库表、索引和 WAL 日志模式
    def _initialize_schema(self):
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(SCHEMA)

    # 返回统一使用的 UTC 时间
    @staticmethod
    def _now():
        return datetime.now(UTC)

    # 将时间转换为可排序的 ISO 8601 文本
    # value：需要持久化的时间
    @staticmethod
    def _serialize_datetime(value):
        return None if value is None else value.astimezone(UTC).isoformat()

    # 将 SQLite 时间文本恢复为 datetime
    # value：数据库保存的 ISO 8601 文本
    @staticmethod
    def _parse_datetime(value):
        return None if value is None else datetime.fromisoformat(value)

    # 计算用于变更检测的稳定文本摘要
    # content：完整记忆文本
    @staticmethod
    def _content_hash(content):
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # 将数据库记录转换为长期记忆模型
    # row：memories 表查询结果
    def _memory_from_row(self, row):
        return LongTermMemory(
            id=row["id"],
            tenant_id=row["tenant_id"],
            user_id=row["user_id"],
            project_id=row["project_id"],
            memory_type=row["memory_type"],
            memory_key=row["memory_key"],
            content=row["content"],
            importance=row["importance"],
            confidence=row["confidence"],
            source_thread_id=row["source_thread_id"],
            source_message_ids=json.loads(row["source_message_ids"]),
            expires_at=self._parse_datetime(row["expires_at"]),
            version=row["version"],
            content_hash=row["content_hash"],
            status=row["status"],
            index_status=row["index_status"],
            embedding_model=row["embedding_model"],
            created_at=self._parse_datetime(row["created_at"]),
            updated_at=self._parse_datetime(row["updated_at"]),
        )

    # 将数据库记录转换为 Outbox 任务模型
    # row：memory_index_jobs 表查询结果
    def _job_from_row(self, row):
        return MemoryIndexJob(
            event_id=row["event_id"],
            memory_id=row["memory_id"],
            memory_version=row["memory_version"],
            operation=row["operation"],
            status=row["status"],
            attempts=row["attempts"],
            next_retry_at=self._parse_datetime(row["next_retry_at"]),
            locked_until=self._parse_datetime(row["locked_until"]),
            last_error=row["last_error"],
            created_at=self._parse_datetime(row["created_at"]),
            updated_at=self._parse_datetime(row["updated_at"]),
        )

    # 在当前事务中创建一条可靠的索引 Outbox 任务
    # connection：当前 SQLite 事务连接
    # memory_id：需要同步到 Qdrant 的记忆标识
    # memory_version：任务对应的记忆版本
    # operation：需要执行的索引操作
    # now：任务创建时间
    def _insert_job(self, connection, memory_id, memory_version, operation, now):
        # event_id：本次 Outbox 事件的唯一标识
        event_id = str(uuid.uuid4())
        # serialized_now：数据库使用的统一时间文本
        serialized_now = self._serialize_datetime(now)
        connection.execute(
            """
            INSERT INTO memory_index_jobs (
                event_id, memory_id, memory_version, operation, status,
                attempts, next_retry_at, locked_until, last_error,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 0, ?, NULL, NULL, ?, ?)
            """,
            (
                event_id,
                memory_id,
                memory_version,
                operation.value,
                IndexJobStatus.PENDING.value,
                serialized_now,
                serialized_now,
                serialized_now,
            ),
        )
        return event_id

    # 新建记忆，或按稳定 memory_key 更新当前有效记忆
    # value：经过 Pydantic 校验的记忆写入请求
    # embedding_model：该记忆准备使用的向量模型
    def upsert(self, value, embedding_model):
        if not isinstance(value, MemoryWrite):
            value = MemoryWrite.model_validate(value)
        if not isinstance(embedding_model, str) or not embedding_model.strip():
            raise ValueError("embedding_model 不能为空")

        # now：本次写入和 Outbox 事件共享的时间
        now = self._now()
        # project_scope：将空项目转换成可参与唯一约束的稳定值
        project_scope = value.project_id or ""
        # content_hash：用于识别记忆文本版本的 SHA-256
        content_hash = self._content_hash(value.content)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            # existing：具有相同稳定键的当前有效记忆
            existing = None
            if value.memory_key is not None:
                existing = connection.execute(
                    """
                    SELECT * FROM memories
                    WHERE tenant_id = ? AND user_id = ? AND project_scope = ?
                      AND memory_type = ? AND memory_key = ? AND status = ?
                    """,
                    (
                        value.tenant_id,
                        value.user_id,
                        project_scope,
                        value.memory_type.value,
                        value.memory_key,
                        MemoryStatus.ACTIVE.value,
                    ),
                ).fetchone()

            if existing is None:
                # memory_id：SQLite 和 Qdrant 共用的新记忆标识
                memory_id = str(uuid.uuid4())
                # memory_version：新记忆从版本 1 开始
                memory_version = 1
                connection.execute(
                    """
                    INSERT INTO memories (
                        id, tenant_id, user_id, project_id, project_scope,
                        memory_type, memory_key, content, content_hash,
                        importance, confidence, status, index_status,
                        embedding_model, version, source_thread_id,
                        source_message_ids, created_at, updated_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        value.tenant_id,
                        value.user_id,
                        value.project_id,
                        project_scope,
                        value.memory_type.value,
                        value.memory_key,
                        value.content,
                        content_hash,
                        value.importance,
                        value.confidence,
                        MemoryStatus.ACTIVE.value,
                        IndexStatus.PENDING.value,
                        embedding_model.strip(),
                        memory_version,
                        value.source_thread_id,
                        json.dumps(value.source_message_ids, ensure_ascii=False),
                        self._serialize_datetime(now),
                        self._serialize_datetime(now),
                        self._serialize_datetime(value.expires_at),
                    ),
                )
            else:
                memory_id = existing["id"]
                memory_version = existing["version"] + 1
                connection.execute(
                    """
                    UPDATE memories
                    SET content = ?, content_hash = ?, importance = ?, confidence = ?,
                        index_status = ?, embedding_model = ?, version = ?,
                        source_thread_id = ?, source_message_ids = ?,
                        updated_at = ?, expires_at = ?
                    WHERE id = ?
                    """,
                    (
                        value.content,
                        content_hash,
                        value.importance,
                        value.confidence,
                        IndexStatus.PENDING.value,
                        embedding_model.strip(),
                        memory_version,
                        value.source_thread_id,
                        json.dumps(value.source_message_ids, ensure_ascii=False),
                        self._serialize_datetime(now),
                        self._serialize_datetime(value.expires_at),
                        memory_id,
                    ),
                )
                # 旧的未领取任务已经不再代表记忆最新版本
                connection.execute(
                    """
                    UPDATE memory_index_jobs
                    SET status = ?, updated_at = ?
                    WHERE memory_id = ? AND status = ?
                    """,
                    (
                        IndexJobStatus.SUPERSEDED.value,
                        self._serialize_datetime(now),
                        memory_id,
                        IndexJobStatus.PENDING.value,
                    ),
                )

            self._insert_job(
                connection,
                memory_id,
                memory_version,
                IndexOperation.UPSERT,
                now,
            )
            # row：提交前重新读取的完整记忆记录
            row = connection.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
            connection.commit()
        return self._memory_from_row(row)

    # 按标识读取一条长期记忆
    # memory_id：记忆唯一标识
    def get(self, memory_id):
        with self._connect() as connection:
            # row：指定标识对应的记忆记录
            row = connection.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return None if row is None else self._memory_from_row(row)

    # 在可信用户范围内按稳定业务键精确读取当前有效记忆
    # tenant_id：经过身份认证的租户标识
    # user_id：经过身份认证的用户标识
    # project_id：记忆所属项目
    # memory_type：长期记忆业务类型
    # memory_key：支持精确查询的稳定键
    def get_by_key(
        self,
        tenant_id,
        user_id,
        project_id,
        memory_type,
        memory_key,
    ):
        # type_value：枚举或字符串形式的记忆类型
        type_value = getattr(memory_type, "value", memory_type)
        with self._connect() as connection:
            # row：指定可信范围和稳定键对应的当前有效记忆
            row = connection.execute(
                """
                SELECT * FROM memories
                WHERE tenant_id = ? AND user_id = ? AND project_scope = ?
                  AND memory_type = ? AND memory_key = ? AND status = ?
                """,
                (
                    tenant_id,
                    user_id,
                    project_id or "",
                    type_value,
                    memory_key,
                    MemoryStatus.ACTIVE.value,
                ),
            ).fetchone()
        return None if row is None else self._memory_from_row(row)

    # 在可信租户和用户范围内批量读取长期记忆
    # memory_ids：Qdrant 返回的记忆标识列表
    # tenant_id：经过身份认证的租户标识
    # user_id：经过身份认证的用户标识
    def get_many(self, memory_ids, tenant_id, user_id):
        if not memory_ids:
            return []
        # unique_ids：保持 Qdrant 排序并去除重复值的记忆标识
        unique_ids = list(dict.fromkeys(memory_ids))
        # placeholders：批量 IN 查询使用的参数占位符
        placeholders = ",".join("?" for _ in unique_ids)
        with self._connect() as connection:
            # rows：经过租户和用户边界校验的有效记忆记录
            rows = connection.execute(
                f"""
                SELECT * FROM memories
                WHERE id IN ({placeholders}) AND tenant_id = ? AND user_id = ?
                  AND status = ? AND index_status = ?
                  AND (expires_at IS NULL OR expires_at > ?)
                """,
                (
                    *unique_ids,
                    tenant_id,
                    user_id,
                    MemoryStatus.ACTIVE.value,
                    IndexStatus.INDEXED.value,
                    self._serialize_datetime(self._now()),
                ),
            ).fetchall()
        # memories_by_id：用于恢复 Qdrant 相关度顺序的记忆映射
        memories_by_id = {row["id"]: self._memory_from_row(row) for row in rows}
        return [memories_by_id[item_id] for item_id in unique_ids if item_id in memories_by_id]

    # 软删除长期记忆并在同一事务中创建 Qdrant 删除任务
    # memory_id：需要删除的记忆标识
    # tenant_id：经过身份认证的租户标识
    # user_id：经过身份认证的用户标识
    def soft_delete(self, memory_id, tenant_id, user_id):
        # now：删除状态与 Outbox 任务共享的时间
        now = self._now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            # row：锁定范围内当前有效的记忆记录
            row = connection.execute(
                """
                SELECT * FROM memories
                WHERE id = ? AND tenant_id = ? AND user_id = ? AND status = ?
                """,
                (memory_id, tenant_id, user_id, MemoryStatus.ACTIVE.value),
            ).fetchone()
            if row is None:
                connection.rollback()
                return False
            # memory_version：删除事件使用的新版本号
            memory_version = row["version"] + 1
            connection.execute(
                """
                UPDATE memories
                SET status = ?, index_status = ?, version = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    MemoryStatus.DELETED.value,
                    IndexStatus.PENDING.value,
                    memory_version,
                    self._serialize_datetime(now),
                    memory_id,
                ),
            )
            connection.execute(
                """
                UPDATE memory_index_jobs
                SET status = ?, updated_at = ?
                WHERE memory_id = ? AND status = ?
                """,
                (
                    IndexJobStatus.SUPERSEDED.value,
                    self._serialize_datetime(now),
                    memory_id,
                    IndexJobStatus.PENDING.value,
                ),
            )
            self._insert_job(
                connection,
                memory_id,
                memory_version,
                IndexOperation.DELETE,
                now,
            )
            connection.commit()
        return True

    # 领取一批到期任务并使用租约防止正常情况下重复消费
    # limit：本批最多领取的任务数量
    # lease_seconds：任务保持 processing 状态的租约秒数
    def claim_jobs(self, limit=16, lease_seconds=120):
        if type(limit) is not int or limit < 1:
            raise ValueError("limit 必须是正整数")
        if type(lease_seconds) is not int or lease_seconds < 1:
            raise ValueError("lease_seconds 必须是正整数")
        # now：任务领取时间
        now = self._now()
        # locked_until：本次领取任务的租约到期时间
        locked_until = now + timedelta(seconds=lease_seconds)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            # 已经过期的 processing 任务重新进入 pending，支持进程崩溃恢复
            connection.execute(
                """
                UPDATE memory_index_jobs
                SET status = ?, locked_until = NULL, updated_at = ?
                WHERE status = ? AND locked_until <= ?
                """,
                (
                    IndexJobStatus.PENDING.value,
                    self._serialize_datetime(now),
                    IndexJobStatus.PROCESSING.value,
                    self._serialize_datetime(now),
                ),
            )
            # rows：按创建顺序选择且不与同一记忆的处理中任务并发
            rows = connection.execute(
                """
                SELECT job.* FROM memory_index_jobs AS job
                WHERE job.status = ? AND job.next_retry_at <= ?
                  AND NOT EXISTS (
                      SELECT 1 FROM memory_index_jobs AS active
                      WHERE active.memory_id = job.memory_id AND active.status = ?
                  )
                ORDER BY job.created_at, job.event_id
                LIMIT ?
                """,
                (
                    IndexJobStatus.PENDING.value,
                    self._serialize_datetime(now),
                    IndexJobStatus.PROCESSING.value,
                    limit,
                ),
            ).fetchall()
            # event_ids：本次成功选择的任务标识
            event_ids = [row["event_id"] for row in rows]
            for event_id in event_ids:
                connection.execute(
                    """
                    UPDATE memory_index_jobs
                    SET status = ?, attempts = attempts + 1,
                        locked_until = ?, updated_at = ?
                    WHERE event_id = ? AND status = ?
                    """,
                    (
                        IndexJobStatus.PROCESSING.value,
                        self._serialize_datetime(locked_until),
                        self._serialize_datetime(now),
                        event_id,
                        IndexJobStatus.PENDING.value,
                    ),
                )
            connection.commit()
            if not event_ids:
                return []
            # placeholders：重新读取已领取任务使用的参数占位符
            placeholders = ",".join("?" for _ in event_ids)
            claimed_rows = connection.execute(
                f"SELECT * FROM memory_index_jobs WHERE event_id IN ({placeholders})",
                event_ids,
            ).fetchall()
        # jobs_by_id：用于保持领取顺序的任务映射
        jobs_by_id = {row["event_id"]: self._job_from_row(row) for row in claimed_rows}
        return [jobs_by_id[event_id] for event_id in event_ids]

    # 将已经可靠同步到 Qdrant 的任务标记为完成
    # event_id：成功任务标识
    def complete_job(self, event_id):
        # now：任务完成时间
        now = self._now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            # job：当前正在处理的 Outbox 任务
            job = connection.execute(
                "SELECT * FROM memory_index_jobs WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if job is None or job["status"] != IndexJobStatus.PROCESSING.value:
                connection.rollback()
                return False
            # memory：用于确认任务仍然对应最新版本的记忆
            memory = connection.execute(
                "SELECT version FROM memories WHERE id = ?",
                (job["memory_id"],),
            ).fetchone()
            # is_current：任务版本是否仍是 SQLite 中的最新记忆版本
            is_current = memory is not None and memory["version"] == job["memory_version"]
            # final_status：旧版本完成后只标记过期，不能修改最新索引状态
            final_status = (
                IndexJobStatus.COMPLETED.value
                if is_current
                else IndexJobStatus.SUPERSEDED.value
            )
            connection.execute(
                """
                UPDATE memory_index_jobs
                SET status = ?, locked_until = NULL, last_error = NULL, updated_at = ?
                WHERE event_id = ?
                """,
                (final_status, self._serialize_datetime(now), event_id),
            )
            if is_current:
                connection.execute(
                    """
                    UPDATE memories SET index_status = ?, updated_at = ? WHERE id = ?
                    """,
                    (
                        IndexStatus.INDEXED.value,
                        self._serialize_datetime(now),
                        job["memory_id"],
                    ),
                )
            connection.commit()
        return is_current

    # 将旧版本任务标记为已被新版本替代
    # event_id：需要跳过的任务标识
    def supersede_job(self, event_id):
        with self._connect() as connection:
            # now：任务被判定过期的时间
            now = self._serialize_datetime(self._now())
            connection.execute(
                """
                UPDATE memory_index_jobs
                SET status = ?, locked_until = NULL, updated_at = ?
                WHERE event_id = ? AND status = ?
                """,
                (
                    IndexJobStatus.SUPERSEDED.value,
                    now,
                    event_id,
                    IndexJobStatus.PROCESSING.value,
                ),
            )

    # 记录索引失败并决定等待重试还是进入死信状态
    # event_id：执行失败的任务标识
    # error：经过清理后可以安全持久化的错误说明
    # retryable：该错误是否允许自动重试
    # max_attempts：任务允许执行的最大次数
    # retry_delay_seconds：下次执行前等待的秒数
    def fail_job(
        self,
        event_id,
        error,
        retryable,
        max_attempts,
        retry_delay_seconds,
    ):
        # now：本次失败发生的时间
        now = self._now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            # job：当前正在处理的任务记录
            job = connection.execute(
                "SELECT * FROM memory_index_jobs WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if job is None or job["status"] != IndexJobStatus.PROCESSING.value:
                connection.rollback()
                return IndexJobStatus.SUPERSEDED
            # memory：用于判断失败任务是否已经落后于记忆最新版本
            memory = connection.execute(
                "SELECT version FROM memories WHERE id = ?",
                (job["memory_id"],),
            ).fetchone()
            if memory is None or memory["version"] != job["memory_version"]:
                # 旧任务无需继续重试，也不能修改最新记忆状态
                connection.execute(
                    """
                    UPDATE memory_index_jobs
                    SET status = ?, locked_until = NULL, last_error = ?, updated_at = ?
                    WHERE event_id = ?
                    """,
                    (
                        IndexJobStatus.SUPERSEDED.value,
                        str(error)[:2000],
                        self._serialize_datetime(now),
                        event_id,
                    ),
                )
                connection.commit()
                return IndexJobStatus.SUPERSEDED

            # should_retry：错误可恢复且任务尚未用完执行次数
            should_retry = retryable and job["attempts"] < max_attempts
            # final_status：失败后重新排队或进入死信状态
            final_status = (
                IndexJobStatus.PENDING if should_retry else IndexJobStatus.DEAD
            )
            # next_retry_at：允许 Worker 再次领取任务的时间
            next_retry_at = (
                now + timedelta(seconds=retry_delay_seconds)
                if should_retry
                else None
            )
            connection.execute(
                """
                UPDATE memory_index_jobs
                SET status = ?, next_retry_at = ?, locked_until = NULL,
                    last_error = ?, updated_at = ?
                WHERE event_id = ?
                """,
                (
                    final_status.value,
                    self._serialize_datetime(next_retry_at),
                    str(error)[:2000],
                    self._serialize_datetime(now),
                    event_id,
                ),
            )
            if final_status is IndexJobStatus.DEAD:
                connection.execute(
                    "UPDATE memories SET index_status = ? WHERE id = ?",
                    (IndexStatus.FAILED.value, job["memory_id"]),
                )
            connection.commit()
        return final_status

    # 读取指定 Outbox 任务，主要用于状态检查和测试
    # event_id：Outbox 任务唯一标识
    def get_job(self, event_id):
        with self._connect() as connection:
            # row：指定标识对应的任务记录
            row = connection.execute(
                "SELECT * FROM memory_index_jobs WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return None if row is None else self._job_from_row(row)

    # 按记忆标识读取 Outbox 任务历史，主要用于监控和故障排查
    # memory_id：可选的记忆唯一标识
    def list_jobs(self, memory_id=None):
        with self._connect() as connection:
            if memory_id is None:
                # rows：全部 Outbox 任务记录
                rows = connection.execute(
                    "SELECT * FROM memory_index_jobs ORDER BY created_at, event_id"
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM memory_index_jobs
                    WHERE memory_id = ? ORDER BY created_at, event_id
                    """,
                    (memory_id,),
                ).fetchall()
        return [self._job_from_row(row) for row in rows]

    # 为当前用户轮次领取一次幂等记忆提取执行权
    # turn_id：当前用户轮次的稳定唯一标识
    # thread_id：当前会话标识
    # user_message_id：触发当前轮次的用户消息标识
    # lease_seconds：进程异常退出后允许重新领取的租约秒数
    def begin_extraction(
        self,
        turn_id,
        thread_id,
        user_message_id,
        lease_seconds=300,
    ):
        # now：本次尝试领取提取任务的时间
        now = self._now()
        # locked_until：本次提取执行权的租约到期时间
        locked_until = now + timedelta(seconds=lease_seconds)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            # row：同一用户轮次已有的提取执行记录
            row = connection.execute(
                "SELECT * FROM memory_extraction_runs WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if row is not None:
                if row["status"] == "completed":
                    connection.rollback()
                    return False
                # active_lease：其他进程仍然有效的提取执行租约
                active_lease = self._parse_datetime(row["locked_until"])
                if (
                    row["status"] == "processing"
                    and active_lease is not None
                    and active_lease > now
                ):
                    connection.rollback()
                    return False
                connection.execute(
                    """
                    UPDATE memory_extraction_runs
                    SET status = 'processing', attempts = attempts + 1,
                        locked_until = ?, last_error = NULL, updated_at = ?
                    WHERE turn_id = ?
                    """,
                    (
                        self._serialize_datetime(locked_until),
                        self._serialize_datetime(now),
                        turn_id,
                    ),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO memory_extraction_runs (
                        turn_id, thread_id, user_message_id, status, attempts,
                        extracted_count, locked_until, last_error,
                        started_at, completed_at, updated_at
                    ) VALUES (?, ?, ?, 'processing', 1, 0, ?, NULL, ?, NULL, ?)
                    """,
                    (
                        turn_id,
                        thread_id,
                        user_message_id,
                        self._serialize_datetime(locked_until),
                        self._serialize_datetime(now),
                        self._serialize_datetime(now),
                    ),
                )
            connection.commit()
        return True

    # 将当前用户轮次标记为已经完成记忆提取
    # turn_id：当前用户轮次的稳定唯一标识
    # extracted_count：本轮实际写入 SQLite 的记忆数量
    def complete_extraction(self, turn_id, extracted_count):
        # now：提取成功完成的时间
        now = self._serialize_datetime(self._now())
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE memory_extraction_runs
                SET status = 'completed', extracted_count = ?, locked_until = NULL,
                    last_error = NULL, completed_at = ?, updated_at = ?
                WHERE turn_id = ? AND status = 'processing'
                """,
                (extracted_count, now, now, turn_id),
            )

    # 记录当前轮次记忆提取失败并释放执行租约
    # turn_id：当前用户轮次的稳定唯一标识
    # error：经过长度限制后可以安全持久化的错误说明
    def fail_extraction(self, turn_id, error):
        # now：提取失败发生的时间
        now = self._serialize_datetime(self._now())
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE memory_extraction_runs
                SET status = 'failed', locked_until = NULL,
                    last_error = ?, updated_at = ?
                WHERE turn_id = ? AND status = 'processing'
                """,
                (str(error)[:2000], now, turn_id),
            )

    # 判断可信用户范围内是否存在可供语义召回的已索引记忆
    # tenant_id：经过身份认证的租户标识
    # user_id：经过身份认证的用户标识
    def has_indexed_memories(self, tenant_id, user_id):
        with self._connect() as connection:
            # row：任意一条仍有效且已经完成向量索引的记忆
            row = connection.execute(
                """
                SELECT 1 FROM memories
                WHERE tenant_id = ? AND user_id = ? AND status = ?
                  AND index_status = ? AND (expires_at IS NULL OR expires_at > ?)
                LIMIT 1
                """,
                (
                    tenant_id,
                    user_id,
                    MemoryStatus.ACTIVE.value,
                    IndexStatus.INDEXED.value,
                    self._serialize_datetime(self._now()),
                ),
            ).fetchone()
        return row is not None

