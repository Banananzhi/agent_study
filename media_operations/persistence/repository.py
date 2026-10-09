"""账号、策略与任务记录。每次操作使用独立连接和短事务。"""

import hashlib
import json
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter

from media_operations.persistence.migrate import migrate, open_database
from media_operations.persistence.errors import ConflictError, NotFoundError
from media_operations.persistence.research_repository import ResearchRepositoryMixin
from media_operations.schemas import (
    AccountBrief, AccountCreate, AccountStatus, AgentRun, AgentTask,
    AgentTaskResult, ContentStrategy, OwnerScope, Payload, RunCreate,
    RunEvent, RunStatus, StrategyCreate, TaskError, TaskStatus,
)


PAYLOAD_ADAPTER = TypeAdapter(Payload)


def _validate(model, value):
    # 重新验证 model_copy(update=...) 等可能绕过校验的模型。
    return model.model_validate(value.model_dump() if isinstance(value, BaseModel) else value)


def _json(value):
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _id(prefix):
    return f"{prefix}_{uuid4().hex}"


def _now():
    return datetime.now(UTC).isoformat()


class MediaRepository(ResearchRepositoryMixin):
    """归属由构造参数绑定；所有 Run/Task 查询还必须提供 account_id。"""

    def __init__(self, path: str | Path = ".agent_data/media_operations.sqlite3", *, owner=None):
        self.path = Path(path)
        self.owner = _validate(OwnerScope, owner or OwnerScope())
        migrate(self.path)

    @contextmanager
    def _connection(self, *, write=False):
        connection = open_database(self.path)
        try:
            # 写事务串行化领取、序号分配及版本校验；模型调用不能在事务中执行。
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _account(self, connection, account_id):
        row = connection.execute(
            "SELECT snapshot_json FROM media_account WHERE account_id=? AND tenant_id=? AND user_id=?",
            (account_id, self.owner.tenant_id, self.owner.user_id),
        ).fetchone()
        if row is None:
            raise NotFoundError("账号不存在或不可访问")
        return AccountBrief.model_validate_json(row["snapshot_json"])

    @staticmethod
    def _strategy(connection, account_id, version=None):
        if version is None:
            row = connection.execute(
                "SELECT snapshot_json FROM content_strategy WHERE account_id=? ORDER BY version DESC LIMIT 1",
                (account_id,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT snapshot_json FROM content_strategy WHERE account_id=? AND version=?", (account_id, version),
            ).fetchone()
        if row is None:
            raise NotFoundError("策略版本不存在")
        return ContentStrategy.model_validate_json(row["snapshot_json"])

    @staticmethod
    def _check_strategy(account, strategy):
        if not set(strategy.pillar_weights) <= set(account.content_pillars):
            raise ConflictError("策略引用了账号中不存在的内容支柱，请先更新策略")

    def _insert_strategy(self, connection, account, config, version):
        self._check_strategy(account, config)
        strategy = ContentStrategy(
            **config.model_dump(), strategy_id=_id("strategy"), account_id=account.account_id,
            version=version, created_at=_now(),
        )
        connection.execute(
            "INSERT INTO content_strategy VALUES (?, ?, ?, ?, ?)",
            (strategy.strategy_id, account.account_id, version, _json(strategy), strategy.created_at.isoformat()),
        )
        for goal in strategy.goals:
            connection.execute(
                "INSERT INTO account_goal VALUES (?, ?, ?, ?, ?, ?, ?)",
                (_id("goal"), strategy.strategy_id, goal.metric, goal.target_value,
                 goal.period_start.isoformat(), goal.period_end.isoformat(), goal.rationale),
            )
        return strategy

    def create_account(self, config: AccountCreate, strategy: StrategyCreate | None = None) -> AccountBrief:
        config = _validate(AccountCreate, config)
        strategy = _validate(StrategyCreate, strategy if strategy is not None else StrategyCreate())
        now = _now()
        account = AccountBrief(
            **config.model_dump(), account_id=_id("account"), version=1, created_at=now, updated_at=now,
        )
        with self._connection(write=True) as connection:
            connection.execute(
                "INSERT INTO media_account VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (account.account_id, self.owner.tenant_id, self.owner.user_id, 1, account.status.value,
                 _json(account), now, now),
            )
            connection.execute("INSERT INTO account_revision VALUES (?, ?, ?, ?)",
                               (account.account_id, 1, _json(account), now))
            self._insert_strategy(connection, account, strategy, 1)
        return account

    def get_account(self, account_id) -> AccountBrief:
        with self._connection() as connection:
            return self._account(connection, account_id)

    def list_accounts(self) -> list[AccountBrief]:
        with self._connection() as connection:
            return [AccountBrief.model_validate_json(row["snapshot_json"]) for row in connection.execute(
                "SELECT snapshot_json FROM media_account WHERE tenant_id=? AND user_id=? ORDER BY created_at, account_id",
                (self.owner.tenant_id, self.owner.user_id),
            )]

    def update_account(self, account_id, config: AccountCreate, *, expected_version: int) -> AccountBrief:
        config = _validate(AccountCreate, config)
        with self._connection(write=True) as connection:
            previous = self._account(connection, account_id)
            if type(expected_version) is not int or previous.version != expected_version:
                raise ConflictError("账号版本已变化")
            now = _now()
            account = AccountBrief(
                **config.model_dump(), account_id=account_id, version=previous.version + 1,
                created_at=previous.created_at, updated_at=now,
            )
            connection.execute(
                "UPDATE media_account SET version=?, status=?, snapshot_json=?, updated_at=? WHERE account_id=?",
                (account.version, account.status.value, _json(account), now, account_id),
            )
            connection.execute("INSERT INTO account_revision VALUES (?, ?, ?, ?)",
                               (account_id, account.version, _json(account), now))
            return account

    def account_history(self, account_id) -> list[AccountBrief]:
        with self._connection() as connection:
            self._account(connection, account_id)
            return [AccountBrief.model_validate_json(row["snapshot_json"]) for row in connection.execute(
                "SELECT snapshot_json FROM account_revision WHERE account_id=? ORDER BY version", (account_id,),
            )]

    def get_strategy(self, account_id, version=None) -> ContentStrategy:
        with self._connection() as connection:
            self._account(connection, account_id)
            return self._strategy(connection, account_id, version)

    def update_strategy(self, account_id, config: StrategyCreate, *, expected_version: int) -> ContentStrategy:
        config = _validate(StrategyCreate, config)
        with self._connection(write=True) as connection:
            account = self._account(connection, account_id)
            previous = self._strategy(connection, account_id)
            if type(expected_version) is not int or previous.version != expected_version:
                raise ConflictError("策略版本已变化")
            return self._insert_strategy(connection, account, config, previous.version + 1)

    def _run(self, connection, account_id, run_id):
        self._account(connection, account_id)
        row = connection.execute(
            "SELECT * FROM agent_run WHERE account_id=? AND run_id=?", (account_id, run_id),
        ).fetchone()
        if row is None:
            raise NotFoundError("Run 不存在或不属于当前账号")
        return AgentRun(
            run_id=row["run_id"], account_id=account_id, request=RunCreate.model_validate_json(row["request_json"]),
            account_snapshot=AccountBrief.model_validate_json(row["account_snapshot_json"]),
            strategy_snapshot=ContentStrategy.model_validate_json(row["strategy_snapshot_json"]),
            status=row["status"], error=TaskError.model_validate_json(row["error_json"]) if row["error_json"] else None,
            version=row["version"], created_at=row["created_at"], updated_at=row["updated_at"],
        )

    @staticmethod
    def _event(connection, run_id, event_type, payload):
        payload = PAYLOAD_ADAPTER.validate_python(payload)
        seq = connection.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM run_event WHERE run_id=?", (run_id,),
        ).fetchone()[0]
        connection.execute("INSERT INTO run_event VALUES (?, ?, ?, ?, ?, ?)",
                           (_id("event"), run_id, seq, event_type, _json(payload), _now()))

    def create_run(self, account_id, request: RunCreate) -> AgentRun:
        request = _validate(RunCreate, request)
        serialized = _json(request)
        fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        with self._connection(write=True) as connection:
            account = self._account(connection, account_id)
            existing = connection.execute(
                "SELECT run_id, request_hash FROM agent_run WHERE account_id=? AND idempotency_key=?",
                (account_id, request.idempotency_key),
            ).fetchone()
            if existing:
                if existing["request_hash"] != fingerprint:
                    raise ConflictError("相同幂等键不能用于不同请求")
                return self._run(connection, account_id, existing["run_id"])
            if account.status != AccountStatus.ACTIVE:
                raise ConflictError("已暂停账号不能创建新 Run")
            strategy = self._strategy(connection, account_id)
            self._check_strategy(account, strategy)
            run_id, now = _id("run"), _now()
            connection.execute("INSERT INTO agent_run VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (run_id, account_id, request.idempotency_key, fingerprint, serialized,
                                _json(account), _json(strategy), RunStatus.PENDING.value, None, 1, now, now))
            task_ids = {task.task_key: _id("task") for task in request.tasks}
            for ordinal, task in enumerate(request.tasks):
                task_id = task_ids[task.task_key]
                connection.execute(
                    "INSERT INTO agent_task VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (task_id, run_id, task.task_key, task.task_type.value, ordinal,
                     _json(task.input), TaskStatus.PENDING.value, 0, None, None, now, now),
                )
                for index, dependency in enumerate(task.depends_on):
                    connection.execute("INSERT INTO task_dependency VALUES (?, ?, ?, ?)",
                                       (run_id, task_id, task_ids[dependency], index))
            self._event(connection, run_id, "run_created", {
                "account_version": account.version, "strategy_version": strategy.version,
                "task_count": len(request.tasks),
            })
            return self._run(connection, account_id, run_id)

    def get_run(self, account_id, run_id) -> AgentRun:
        with self._connection() as connection:
            return self._run(connection, account_id, run_id)

    def list_runs(self, account_id) -> list[AgentRun]:
        with self._connection() as connection:
            self._account(connection, account_id)
            ids = [row[0] for row in connection.execute(
                "SELECT run_id FROM agent_run WHERE account_id=? ORDER BY created_at, run_id", (account_id,),
            )]
            return [self._run(connection, account_id, run_id) for run_id in ids]

    @staticmethod
    def _task(connection, run_id, task_id):
        row = connection.execute("SELECT * FROM agent_task WHERE run_id=? AND task_id=?",
                                 (run_id, task_id)).fetchone()
        if row is None:
            raise NotFoundError("Task 不存在或不属于当前 Run")
        dependencies = [item[0] for item in connection.execute(
            "SELECT dependency_id FROM task_dependency WHERE task_id=? ORDER BY ordinal", (task_id,),
        )]
        return AgentTask(
            task_id=task_id, run_id=run_id, task_key=row["task_key"], task_type=row["task_type"],
            dependencies=dependencies, input=PAYLOAD_ADAPTER.validate_json(row["input_json"]),
            status=row["status"], attempts=row["attempts"],
            result=AgentTaskResult[Payload].model_validate_json(row["result_json"]) if row["result_json"] else None,
            skip_reason=row["skip_reason"], created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def get_task(self, account_id, run_id, task_id) -> AgentTask:
        with self._connection() as connection:
            self._run(connection, account_id, run_id)
            return self._task(connection, run_id, task_id)

    def list_tasks(self, account_id, run_id) -> list[AgentTask]:
        with self._connection() as connection:
            self._run(connection, account_id, run_id)
            return [self._task(connection, run_id, row[0]) for row in connection.execute(
                "SELECT task_id FROM agent_task WHERE run_id=? ORDER BY ordinal", (run_id,),
            )]

    def list_events(self, account_id, run_id, *, after_seq=0, limit=100) -> list[RunEvent]:
        if type(after_seq) is not int or after_seq < 0 or type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("事件游标必须非负，分页大小为 1–1000")
        with self._connection() as connection:
            self._run(connection, account_id, run_id)
            return [RunEvent(
                event_id=row["event_id"], run_id=run_id, seq=row["seq"], event_type=row["event_type"],
                payload=PAYLOAD_ADAPTER.validate_json(row["payload_json"]), created_at=row["created_at"],
            ) for row in connection.execute(
                "SELECT * FROM run_event WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?", (run_id, after_seq, limit),
            )]

    def _change_run(self, connection, run, status, event_type, error=None):
        connection.execute(
            "UPDATE agent_run SET status=?, error_json=?, version=version+1, updated_at=? WHERE run_id=?",
            (status.value, _json(error) if error else None, _now(), run.run_id),
        )
        self._event(connection, run.run_id, event_type, {
            "previous_status": run.status.value, "status": status.value,
            **({"error_code": error.code} if error else {}),
        })

    def start_run(self, account_id, run_id) -> AgentRun:
        with self._connection(write=True) as connection:
            run = self._run(connection, account_id, run_id)
            if run.status != RunStatus.PENDING:
                raise ConflictError("只有 PENDING Run 可以开始")
            if self._account(connection, account_id).status != AccountStatus.ACTIVE:
                raise ConflictError("账号已暂停")
            self._change_run(connection, run, RunStatus.RUNNING, "run_started")
            return self._run(connection, account_id, run_id)

    def claim_next_run(self) -> AgentRun | None:
        """为后续单 Worker 提供原子领取原语；此方法不执行 Agent。"""
        with self._connection(write=True) as connection:
            row = connection.execute("""
                SELECT r.run_id, r.account_id FROM agent_run r
                JOIN media_account a ON r.account_id=a.account_id
                WHERE a.tenant_id=? AND a.user_id=? AND a.status='ACTIVE' AND r.status='PENDING'
                ORDER BY r.created_at, r.run_id LIMIT 1
            """, (self.owner.tenant_id, self.owner.user_id)).fetchone()
            if row is None:
                return None
            run = self._run(connection, row["account_id"], row["run_id"])
            self._change_run(connection, run, RunStatus.RUNNING, "run_started")
            return self._run(connection, row["account_id"], row["run_id"])

    def start_task(self, account_id, run_id, task_id) -> AgentTask:
        with self._connection(write=True) as connection:
            run = self._run(connection, account_id, run_id)
            task = self._task(connection, run_id, task_id)
            if run.status != RunStatus.RUNNING or task.status not in {TaskStatus.PENDING, TaskStatus.READY}:
                raise ConflictError("当前 Run/Task 状态不允许开始任务")
            if connection.execute("SELECT 1 FROM agent_task WHERE run_id=? AND status='RUNNING'", (run_id,)).fetchone():
                raise ConflictError("第一版每个 Run 同时只能执行一个阶段")
            if any(self._task(connection, run_id, dep).status != TaskStatus.COMPLETED for dep in task.dependencies):
                raise ConflictError("前置任务尚未完成")
            if task.status == TaskStatus.PENDING:
                connection.execute("UPDATE agent_task SET status='READY', updated_at=? WHERE task_id=?", (_now(), task_id))
                self._event(connection, run_id, "task_ready", {"task_id": task_id})
            connection.execute(
                "UPDATE agent_task SET status='RUNNING', attempts=attempts+1, updated_at=? WHERE task_id=?", (_now(), task_id),
            )
            self._event(connection, run_id, "task_started", {"task_id": task_id, "task_type": task.task_type.value})
            return self._task(connection, run_id, task_id)

    def complete_task(self, account_id, run_id, task_id, result: AgentTaskResult[Payload]) -> AgentTask:
        result = _validate(AgentTaskResult[Payload], result)
        if not result.success:
            raise ValueError("失败结果必须通过 fail_task 提交")
        with self._connection(write=True) as connection:
            run = self._run(connection, account_id, run_id)
            task = self._task(connection, run_id, task_id)
            if run.status != RunStatus.RUNNING or task.status != TaskStatus.RUNNING:
                raise ConflictError("任务不再运行，不能提交结果")
            connection.execute(
                "UPDATE agent_task SET status='COMPLETED', result_json=?, updated_at=? WHERE task_id=?",
                (_json(result), _now(), task_id),
            )
            self._event(connection, run_id, "task_completed", {"task_id": task_id})
            return self._task(connection, run_id, task_id)

    def _stop_tasks(self, connection, run_id, error):
        now = _now()
        for row in connection.execute(
            "SELECT task_id, status FROM agent_task WHERE run_id=? AND status IN ('PENDING', 'READY', 'RUNNING') ORDER BY ordinal",
            (run_id,),
        ).fetchall():
            task_id = row["task_id"]
            if row["status"] == TaskStatus.RUNNING.value:
                result = AgentTaskResult[Payload](success=False, error=error)
                connection.execute(
                    "UPDATE agent_task SET status='FAILED', result_json=?, updated_at=? WHERE task_id=?", (_json(result), now, task_id),
                )
                self._event(connection, run_id, "task_failed", {"task_id": task_id, "error_code": error.code})
            else:
                connection.execute(
                    "UPDATE agent_task SET status='SKIPPED', skip_reason=?, updated_at=? WHERE task_id=?", (error.message, now, task_id),
                )
                self._event(connection, run_id, "task_skipped", {"task_id": task_id, "error_code": error.code})

    def fail_task(self, account_id, run_id, task_id, error: TaskError) -> AgentRun:
        error = _validate(TaskError, error)
        with self._connection(write=True) as connection:
            run = self._run(connection, account_id, run_id)
            task = self._task(connection, run_id, task_id)
            if run.status != RunStatus.RUNNING or task.status != TaskStatus.RUNNING:
                raise ConflictError("只有运行中的任务可以失败")
            self._stop_tasks(connection, run_id, error)
            self._change_run(connection, run, RunStatus.FAILED, "run_failed", error)
            return self._run(connection, account_id, run_id)

    def finish_run(self, account_id, run_id) -> AgentRun:
        """阶段完成只进入待人工审批；1A 不提供批准或发布接口。"""
        with self._connection(write=True) as connection:
            run = self._run(connection, account_id, run_id)
            if run.status != RunStatus.RUNNING:
                raise ConflictError("只有运行中的 Run 可以完成阶段执行")
            unfinished = connection.execute(
                "SELECT 1 FROM agent_task WHERE run_id=? AND status!='COMPLETED'", (run_id,),
            ).fetchone()
            if unfinished:
                raise ConflictError("仍有未完成任务")
            self._change_run(connection, run, RunStatus.WAITING_APPROVAL, "run_waiting_approval")
            return self._run(connection, account_id, run_id)

    def cancel_run(self, account_id, run_id) -> AgentRun:
        with self._connection(write=True) as connection:
            run = self._run(connection, account_id, run_id)
            if run.status == RunStatus.CANCELLED:
                return run
            if run.status not in {RunStatus.PENDING, RunStatus.RUNNING, RunStatus.WAITING_APPROVAL}:
                raise ConflictError("Run 已结束，不能取消")
            error = TaskError(code="CANCELLED", message="用户取消任务")
            self._stop_tasks(connection, run_id, error)
            self._change_run(connection, run, RunStatus.CANCELLED, "run_cancelled", error)
            return self._run(connection, account_id, run_id)

    def mark_interrupted_runs(self) -> int:
        """未来 Worker 独占启动时显式调用；查询/重新打开库不自动改状态。"""
        with self._connection(write=True) as connection:
            rows = connection.execute("""
                SELECT r.account_id, r.run_id FROM agent_run r
                JOIN media_account a ON a.account_id=r.account_id
                WHERE a.tenant_id=? AND a.user_id=? AND r.status='RUNNING'
            """, (self.owner.tenant_id, self.owner.user_id)).fetchall()
            error = TaskError(code="INTERRUPTED", message="执行进程中断，需显式新建任务重跑")
            for row in rows:
                run = self._run(connection, row["account_id"], row["run_id"])
                self._stop_tasks(connection, run.run_id, error)
                self._change_run(connection, run, RunStatus.FAILED, "run_interrupted", error)
            return len(rows)

