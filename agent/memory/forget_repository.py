import json


FORGET_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL, user_id TEXT NOT NULL, project_id TEXT,
    kind TEXT NOT NULL, turn_id TEXT NOT NULL, payload TEXT NOT NULL,
    UNIQUE(tenant_id, user_id, kind, turn_id)
);
CREATE TABLE IF NOT EXISTS memory_thread_owners (
    thread_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, user_id TEXT NOT NULL,
    project_id TEXT
);
"""


class ForgetRepositoryMixin:
    # 旧 Worker 在遗忘后才完成远程写入时，重新排队最新版本以修复索引
    # job：已执行过远程操作的旧任务
    def repair_stale_index(self, job):
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            # current：SQLite 始终作为最终事实来源
            current = connection.execute("SELECT version FROM memories WHERE id=?", (job.memory_id,)).fetchone()
            if current is None or current["version"] == job.memory_version:
                return
            connection.execute(
                """UPDATE memory_index_jobs SET status='pending', attempts=0,
                locked_until=NULL,next_retry_at=?,updated_at=?
                WHERE memory_id=? AND memory_version=?""",
                (self._serialize_datetime(self._now()), self._serialize_datetime(self._now()),
                 job.memory_id, current["version"]),
            )
            connection.execute("UPDATE memories SET index_status='pending' WHERE id=?", (job.memory_id,))

    # 为会话绑定可信身份，避免通过复用 thread_id 读取他人检查点
    # scope：可信身份字典；thread_id：持久化会话标识
    def bind_memory_thread(self, scope, thread_id):
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT OR IGNORE INTO memory_thread_owners VALUES (?, ?, ?, ?)",
                (thread_id, scope["tenant_id"], scope["user_id"], scope.get("project_id")),
            )
            # row：已登记的会话归属
            row = connection.execute("SELECT * FROM memory_thread_owners WHERE thread_id=?", (thread_id,)).fetchone()
            if (row["tenant_id"], row["user_id"], row["project_id"]) != (
                scope["tenant_id"], scope["user_id"], scope.get("project_id")
            ):
                raise PermissionError("会话不属于当前用户或项目")

    # 登记当前用户轮次，幂等返回数据库生成的来源序号
    # scope：可信身份；turn_id：本轮稳定标识
    def register_memory_turn(self, scope, turn_id):
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO memory_events(tenant_id,user_id,project_id,kind,turn_id,payload) VALUES (?,?,?,'turn',?,'{}')",
                (scope["tenant_id"], scope["user_id"], scope.get("project_id"), turn_id),
            )
            return connection.execute(
                "SELECT seq FROM memory_events WHERE tenant_id=? AND user_id=? AND kind='turn' AND turn_id=?",
                (scope["tenant_id"], scope["user_id"], turn_id),
            ).fetchone()[0]

    # 读取当前项目及用户级的遗忘事件，旧会话恢复和写入门禁共用
    # scope：可信身份；connection：可选事务连接
    def forget_events(self, scope, connection=None):
        from contextlib import nullcontext
        with self._connect() if connection is None else nullcontext(connection) as connection:
            # rows：按事件先后排列的遗忘记录
            rows = connection.execute(
                """SELECT * FROM memory_events WHERE tenant_id=? AND user_id=? AND kind='forget'
                AND (project_id IS NULL OR project_id=?) ORDER BY seq""",
                (scope["tenant_id"], scope["user_id"], scope.get("project_id")),
            ).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    # 原子登记遗忘事件并软删除明确匹配的记忆，整个批次版本不符则回滚
    # scope：可信身份；turn_id：本轮标识；topic：不包含旧值的遗忘主题
    # targets：本轮搜索提供的 id/version 列表
    def commit_forget(self, scope, turn_id, topic, targets):
        from agent.memory.models import IndexOperation
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            # existing：重试相同操作时返回原事件，不重复删除或新增 Outbox
            existing = connection.execute(
                "SELECT * FROM memory_events WHERE tenant_id=? AND user_id=? AND kind='forget' AND turn_id=?",
                (scope["tenant_id"], scope["user_id"], turn_id),
            ).fetchone()
            if existing is not None:
                if existing["project_id"] != scope.get("project_id"):
                    raise ValueError("遗忘操作标识不属于当前项目")
                return existing["seq"]
            # now：遗忘事务统一时间；rows：校验通过后才统一执行的旧记录
            now = self._serialize_datetime(self._now())
            rows = []
            for target in targets:
                # row：必须属于当前精确项目和可信身份的目标
                row = connection.execute(
                    """SELECT * FROM memories WHERE id=? AND tenant_id=? AND user_id=?
                    AND project_scope=?""",
                    (target["id"], scope["tenant_id"], scope["user_id"], scope.get("project_id") or ""),
                ).fetchone()
                if row is None or (row["status"] != "deleted" and row["version"] != target["version"]):
                    raise ValueError("遗忘目标已变化或不属于当前范围，请重新搜索")
                if row["status"] != "deleted":
                    rows.append(row)
            # payload：保留主题与 ID，不重复保存旧正文
            payload = json.dumps({"topic": topic, "memory_ids": [item["id"] for item in targets]}, ensure_ascii=False)
            cursor = connection.execute(
                "INSERT INTO memory_events(tenant_id,user_id,project_id,kind,turn_id,payload) VALUES (?,?,?,'forget',?,?)",
                (scope["tenant_id"], scope["user_id"], scope.get("project_id"), turn_id, payload),
            )
            # event_seq：此次遗忘的全局单调序号
            event_seq = cursor.lastrowid
            for row in rows:
                connection.execute(
                    "INSERT OR IGNORE INTO memory_versions VALUES (?,?,?,?)",
                    (row["id"], row["version"], json.dumps(dict(row), ensure_ascii=False), now),
                )
                connection.execute(
                    "UPDATE memories SET status='deleted',index_status='pending',version=version+1,updated_at=? WHERE id=?",
                    (now, row["id"]),
                )
                connection.execute(
                    "UPDATE memory_index_jobs SET status='superseded',updated_at=? WHERE memory_id=? AND status='pending'",
                    (now, row["id"]),
                )
                self._insert_job(connection, row["id"], row["version"] + 1, IndexOperation.DELETE, self._now())
        return event_seq
