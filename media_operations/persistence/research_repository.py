"""研究存储扩展，与账号/任务 Repository 共用事务和可信归属边界。"""

import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import TypeAdapter

from media_operations.persistence.errors import ConflictError, NotFoundError
from media_operations.research_models import SourceSnapshot, ToolExecution, validate_tool_receipt
from media_operations.schemas import Payload, RunStatus, TaskStatus, TaskType


def _dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class ResearchRepositoryMixin:
    def _active_research_task(self, connection, account_id, run_id, task_id):
        run = self._run(connection, account_id, run_id)
        task = self._task(connection, run_id, task_id)
        if run.status != RunStatus.RUNNING or task.status != TaskStatus.RUNNING or task.task_type != TaskType.RESEARCH:
            raise ConflictError("研究工具只允许在运行中的 research Task 使用")

    def save_sources(self, account_id, run_id, task_id, snapshots: list[SourceSnapshot]) -> list[SourceSnapshot]:
        snapshots = [SourceSnapshot.model_validate(item.model_dump()) for item in snapshots]
        if any((item.account_id, item.run_id, item.task_id) != (account_id, run_id, task_id) for item in snapshots):
            raise NotFoundError("来源不属于当前账号/Run/Task")
        with self._connection(write=True) as connection:
            self._active_research_task(connection, account_id, run_id, task_id)
            stored = []
            for item in snapshots:
                source = item.source
                fingerprint = hashlib.sha256(_dump({
                    "kind": source.kind.value, "provider": source.provider,
                    "url": source.original_url, "final_url": source.final_url,
                    "path": source.relative_path, "hash": source.content_hash,
                    "title": source.title, "published_at_raw": source.published_at_raw,
                    "evidence_start": source.evidence_start,
                }).encode("utf-8")).hexdigest()
                row = connection.execute(
                    "SELECT snapshot_json FROM research_source WHERE task_id=? AND fingerprint=?", (task_id, fingerprint),
                ).fetchone()
                if row:
                    existing = SourceSnapshot.model_validate_json(row[0])
                    if (existing.account_id, existing.run_id, existing.task_id) != (account_id, run_id, task_id):
                        raise NotFoundError("已保存来源不属于当前账号/Run/Task")
                    stored.append(existing)
                else:
                    connection.execute("INSERT INTO research_source VALUES (?, ?, ?, ?, ?, ?)", (
                        source.source_id, run_id, task_id, fingerprint,
                        item.model_dump_json(), source.retrieved_at.isoformat(),
                    ))
                    stored.append(item)
            if snapshots:
                self._event(connection, run_id, "research_sources_saved", {
                    "task_id": task_id, "source_ids": [item.source.source_id for item in stored],
                })
            return stored

    def get_source(self, account_id, run_id, source_id) -> SourceSnapshot:
        with self._connection() as connection:
            self._run(connection, account_id, run_id)
            row = connection.execute(
                "SELECT snapshot_json FROM research_source WHERE run_id=? AND source_id=?", (run_id, source_id),
            ).fetchone()
            if row is None:
                raise NotFoundError("来源不存在或不属于当前 Run")
            snapshot = SourceSnapshot.model_validate_json(row[0])
            if (snapshot.account_id, snapshot.run_id, snapshot.source.source_id) != (account_id, run_id, source_id):
                raise NotFoundError("来源不属于当前账号/Run")
            return snapshot

    def list_sources(self, account_id, run_id) -> list[SourceSnapshot]:
        with self._connection() as connection:
            self._run(connection, account_id, run_id)
            snapshots = [SourceSnapshot.model_validate_json(row[0]) for row in connection.execute(
                "SELECT snapshot_json FROM research_source WHERE run_id=? ORDER BY created_at, source_id", (run_id,),
            )]
            if any((item.account_id, item.run_id) != (account_id, run_id) for item in snapshots):
                raise NotFoundError("来源不属于当前账号/Run")
            return snapshots

    def begin_tool_execution(self, account_id, run_id, task_id, tool_name, arguments: Payload) -> str:
        arguments = TypeAdapter(Payload).validate_python(arguments)
        execution_id = f"execution_{uuid4().hex}"
        record = ToolExecution(
            execution_id=execution_id, run_id=run_id, task_id=task_id, tool_name=tool_name,
            arguments=arguments, status="RUNNING", attempts=0, duration_ms=0, created_at=datetime.now(UTC),
        )
        with self._connection(write=True) as connection:
            self._active_research_task(connection, account_id, run_id, task_id)
            connection.execute("INSERT INTO tool_execution VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                execution_id, run_id, task_id, tool_name, _dump(record.arguments), "RUNNING", None, None,
                0, 0, record.created_at.isoformat(), None,
            ))
            self._event(connection, run_id, "tool_started", {"task_id": task_id, "execution_id": execution_id, "tool_name": tool_name})
        return execution_id

    def finish_tool_execution(self, account_id, run_id, execution_id, result: Payload):
        result = TypeAdapter(Payload).validate_python(result)
        validate_tool_receipt(result)
        with self._connection(write=True) as connection:
            self._run(connection, account_id, run_id)
            row = connection.execute(
                "SELECT * FROM tool_execution WHERE run_id=? AND execution_id=?", (run_id, execution_id),
            ).fetchone()
            if row is None:
                raise NotFoundError("工具记录不属于当前 Run")
            if row["status"] != "RUNNING":
                raise ConflictError("工具记录已结束")
            error_code = result.get("error", {}).get("code") if not result["ok"] else None
            record = ToolExecution(
                execution_id=execution_id, run_id=run_id, task_id=row["task_id"], tool_name=row["tool_name"],
                arguments=json.loads(row["arguments_json"]), status="SUCCEEDED" if result["ok"] else "FAILED",
                result=result, error_code=error_code, attempts=result.get("attempts", 0),
                duration_ms=result.get("duration_ms", 0), created_at=row["created_at"], finished_at=datetime.now(UTC),
            )
            connection.execute("""UPDATE tool_execution SET status=?, result_json=?, error_code=?,
                attempts=?, duration_ms=?, finished_at=? WHERE execution_id=?""", (
                record.status, _dump(result), error_code, record.attempts, record.duration_ms,
                record.finished_at.isoformat(), execution_id,
            ))
            self._event(connection, run_id, "tool_finished", {
                "execution_id": execution_id, "task_id": record.task_id, "status": record.status,
                "attempts": record.attempts, "error_code": error_code,
            })

    def list_tool_executions(self, account_id, run_id) -> list[ToolExecution]:
        with self._connection() as connection:
            self._run(connection, account_id, run_id)
            return [ToolExecution(
                execution_id=row["execution_id"], run_id=run_id, task_id=row["task_id"], tool_name=row["tool_name"],
                arguments=json.loads(row["arguments_json"]), status=row["status"],
                result=json.loads(row["result_json"]) if row["result_json"] else None,
                error_code=row["error_code"], attempts=row["attempts"], duration_ms=row["duration_ms"],
                created_at=row["created_at"], finished_at=row["finished_at"],
            ) for row in connection.execute(
                "SELECT * FROM tool_execution WHERE run_id=? ORDER BY created_at, execution_id", (run_id,),
            )]
