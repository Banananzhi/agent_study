import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from media_operations.persistence.repository import ConflictError, MediaRepository, NotFoundError
from media_operations.schemas import (
    AccountCreate, AccountStatus, AgentTaskResult, OwnerScope, Payload,
    RunCreate, RunStatus, StrategyCreate, TaskError, TaskStatus,
)


def account_config(**changes):
    return AccountCreate(**{
        "account_name": "AI 技术分享", "positioning": "AI Agent 开发实战",
        "target_audience": "Java 开发者", "tone": "准确、清晰",
        "content_pillars": ["教程", "实战"], **changes,
    })


class MediaRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "media.sqlite3"
        self.repository = MediaRepository(self.path)
        self.account = self.repository.create_account(account_config())
        self.account_id = self.account.account_id

    def new_run(self, key="run-key"):
        return self.repository.create_run(self.account_id, RunCreate(idempotency_key=key, goal="制作 MCP 教程"))

    def result(self, stage="research"):
        return AgentTaskResult[Payload](success=True, data={"simulation": True, "stage": stage, "items": [1, "资料"]})

    def complete_first(self, run):
        task = self.repository.list_tasks(self.account_id, run.run_id)[0]
        self.repository.start_task(self.account_id, run.run_id, task.task_id)
        self.repository.complete_task(self.account_id, run.run_id, task.task_id, self.result())
        return task

    def test_account_and_strategy_versions_preserve_history_and_run_snapshots(self):
        run = self.new_run()
        updated = self.repository.update_account(self.account_id, account_config(tone="简洁"), expected_version=1)
        strategy = self.repository.update_strategy(
            self.account_id, StrategyCreate(quality_rules=["引用官方文档"], rationale="用户更新"), expected_version=1,
        )
        self.assertEqual(updated.version, 2)
        self.assertEqual([item.tone for item in self.repository.account_history(self.account_id)], ["准确、清晰", "简洁"])
        self.assertEqual(self.repository.get_strategy(self.account_id, 1).quality_rules, [])
        old = self.repository.get_run(self.account_id, run.run_id)
        self.assertEqual((old.account_snapshot.version, old.strategy_snapshot.version), (1, 1))
        new = self.new_run("second")
        self.assertEqual((new.account_snapshot.version, new.strategy_snapshot.version), (2, strategy.version))

    def test_stale_account_and_strategy_updates_do_not_overwrite(self):
        self.repository.update_account(self.account_id, account_config(tone="新风格"), expected_version=1)
        with self.assertRaises(ConflictError):
            self.repository.update_account(self.account_id, account_config(tone="过期风格"), expected_version=1)
        self.repository.update_strategy(self.account_id, StrategyCreate(rationale="新策略"), expected_version=1)
        with self.assertRaises(ConflictError):
            self.repository.update_strategy(self.account_id, StrategyCreate(rationale="过期策略"), expected_version=1)
        self.assertEqual(self.repository.get_account(self.account_id).tone, "新风格")
        self.assertEqual(self.repository.get_strategy(self.account_id).rationale, "新策略")

    def test_account_and_goals_are_created_atomically(self):
        strategy = StrategyCreate(goals=[{
            "metric": "每周篇数", "target_value": 3, "period_start": "2026-10-09",
            "period_end": "2026-10-16", "rationale": "示例目标",
        }])
        account = self.repository.create_account(account_config(), strategy)
        self.assertEqual(self.repository.get_strategy(account.account_id).goals[0].target_value, 3)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM account_goal").fetchone()[0], 1)
        before = len(self.repository.list_accounts())
        with self.assertRaises(ConflictError):
            self.repository.create_account(account_config(), StrategyCreate(pillar_weights={"陌生支柱": 1}))
        self.assertEqual(len(self.repository.list_accounts()), before)

    def test_paused_accounts_cannot_enqueue_or_start(self):
        run = self.new_run()
        self.repository.update_account(self.account_id, account_config(status=AccountStatus.PAUSED), expected_version=1)
        with self.assertRaises(ConflictError):
            self.new_run("new-key")
        with self.assertRaises(ConflictError):
            self.repository.start_run(self.account_id, run.run_id)
        self.assertIsNone(self.repository.claim_next_run())
        self.assertEqual(self.new_run().run_id, run.run_id)  # 重复请求仍可查询原任务。

    def test_existing_strategy_must_match_current_pillars_before_new_run(self):
        self.repository.update_strategy(self.account_id, StrategyCreate(pillar_weights={"教程": 1}), expected_version=1)
        self.repository.update_account(self.account_id, account_config(content_pillars=["新方向"]), expected_version=1)
        with self.assertRaises(ConflictError):
            self.new_run()
        self.repository.update_strategy(self.account_id, StrategyCreate(pillar_weights={"新方向": 1}), expected_version=2)
        self.assertEqual(self.new_run().account_snapshot.content_pillars, ["新方向"])

    def test_idempotency_conflicts_and_account_isolation(self):
        run = self.new_run()
        self.assertEqual(run, self.new_run())
        with self.assertRaises(ConflictError):
            self.repository.create_run(self.account_id, RunCreate(idempotency_key="run-key", goal="不同目标"))
        another = self.repository.create_account(account_config(account_name="另一个账号"))
        other_run = self.repository.create_run(another.account_id, run.request)
        self.assertNotEqual(other_run.run_id, run.run_id)
        for action in (
            lambda: self.repository.get_run(another.account_id, run.run_id),
            lambda: self.repository.list_tasks(another.account_id, run.run_id),
            lambda: self.repository.list_events(another.account_id, run.run_id),
        ):
            with self.assertRaises(NotFoundError):
                action()
        task = self.repository.list_tasks(self.account_id, run.run_id)[0]
        with self.assertRaises(NotFoundError):
            self.repository.get_task(another.account_id, other_run.run_id, task.task_id)

    def test_owner_scope_is_enforced_on_reads_writes_and_claims(self):
        run = self.new_run()
        other = MediaRepository(self.path, owner=OwnerScope(user_id="other-user"))
        self.assertEqual(other.list_accounts(), [])
        self.assertIsNone(other.claim_next_run())
        for action in (
            lambda: other.get_account(self.account_id),
            lambda: other.get_strategy(self.account_id),
            lambda: other.account_history(self.account_id),
            lambda: other.list_runs(self.account_id),
            lambda: other.cancel_run(self.account_id, run.run_id),
            lambda: other.update_strategy(self.account_id, StrategyCreate(), expected_version=1),
        ):
            with self.assertRaises(NotFoundError):
                action()
        other_account = other.create_account(account_config())
        self.assertEqual([account.account_id for account in other.list_accounts()], [other_account.account_id])

    def test_concurrent_duplicate_submissions_and_claims_are_atomic(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            runs = list(pool.map(lambda _: self.new_run(), range(4)))
        self.assertEqual(len({run.run_id for run in runs}), 1)
        self.assertEqual(len(self.repository.list_tasks(self.account_id, runs[0].run_id)), 6)
        with ThreadPoolExecutor(max_workers=4) as pool:
            claimed = list(pool.map(lambda _: self.repository.claim_next_run(), range(4)))
        self.assertEqual(sum(run is not None for run in claimed), 1)
        self.assertEqual(len(self.repository.list_events(self.account_id, runs[0].run_id)), 2)

    def test_ordered_task_execution_and_cannot_finish_early(self):
        run = self.new_run()
        tasks = self.repository.list_tasks(self.account_id, run.run_id)
        with self.assertRaises(ConflictError):
            self.repository.start_task(self.account_id, run.run_id, tasks[0].task_id)
        self.repository.start_run(self.account_id, run.run_id)
        with self.assertRaises(ConflictError):
            self.repository.start_task(self.account_id, run.run_id, tasks[1].task_id)
        with self.assertRaises(ConflictError):
            self.repository.finish_run(self.account_id, run.run_id)
        for task in tasks:
            started = self.repository.start_task(self.account_id, run.run_id, task.task_id)
            self.assertEqual(started.attempts, 1)
            self.repository.complete_task(self.account_id, run.run_id, task.task_id, self.result(task.task_key))
        finished = self.repository.finish_run(self.account_id, run.run_id)
        self.assertEqual(finished.status, RunStatus.WAITING_APPROVAL)
        self.assertNotEqual(finished.status, RunStatus.COMPLETED)

    def test_duplicate_task_start_and_result_submission_are_rejected(self):
        run = self.new_run()
        self.repository.start_run(self.account_id, run.run_id)
        task = self.complete_first(run)
        for action in (
            lambda: self.repository.start_task(self.account_id, run.run_id, task.task_id),
            lambda: self.repository.complete_task(self.account_id, run.run_id, task.task_id, self.result()),
        ):
            with self.assertRaises(ConflictError):
                action()

    def test_failure_keeps_prior_results_and_skips_downstream(self):
        run = self.new_run()
        self.repository.start_run(self.account_id, run.run_id)
        self.complete_first(run)
        tasks = self.repository.list_tasks(self.account_id, run.run_id)
        self.repository.start_task(self.account_id, run.run_id, tasks[1].task_id)
        failed = self.repository.fail_task(self.account_id, run.run_id, tasks[1].task_id,
                                           TaskError(code="SEARCH_UNAVAILABLE", message="无法检索"))
        tasks = self.repository.list_tasks(self.account_id, run.run_id)
        self.assertEqual(failed.status, RunStatus.FAILED)
        self.assertEqual([task.status for task in tasks], [TaskStatus.COMPLETED, TaskStatus.FAILED] + [TaskStatus.SKIPPED] * 4)
        self.assertEqual(tasks[0].result.data, self.result().data)
        self.assertEqual(tasks[1].result.error.code, "SEARCH_UNAVAILABLE")
        with self.assertRaises(ConflictError):
            self.repository.cancel_run(self.account_id, run.run_id)

    def test_cancel_rejects_late_results_and_is_idempotent(self):
        run = self.new_run()
        self.repository.start_run(self.account_id, run.run_id)
        task = self.repository.list_tasks(self.account_id, run.run_id)[0]
        self.repository.start_task(self.account_id, run.run_id, task.task_id)
        cancelled = self.repository.cancel_run(self.account_id, run.run_id)
        self.assertEqual(cancelled.status, RunStatus.CANCELLED)
        count = len(self.repository.list_events(self.account_id, run.run_id))
        self.repository.cancel_run(self.account_id, run.run_id)
        self.assertEqual(len(self.repository.list_events(self.account_id, run.run_id)), count)
        with self.assertRaises(ConflictError):
            self.repository.complete_task(self.account_id, run.run_id, task.task_id, self.result())
        self.assertEqual(self.repository.get_task(self.account_id, run.run_id, task.task_id).result.error.code, "CANCELLED")

    def test_cancel_pending_skips_all_tasks(self):
        run = self.new_run()
        self.repository.cancel_run(self.account_id, run.run_id)
        self.assertTrue(all(task.status == TaskStatus.SKIPPED for task in self.repository.list_tasks(self.account_id, run.run_id)))

    def test_reopen_preserves_results_events_and_snapshots(self):
        run = self.new_run()
        self.repository.start_run(self.account_id, run.run_id)
        task = self.complete_first(run)
        reopened = MediaRepository(self.path)
        self.assertEqual(reopened.get_task(self.account_id, run.run_id, task.task_id).result, self.result())
        self.assertEqual(reopened.get_run(self.account_id, run.run_id).status, RunStatus.RUNNING)
        self.assertEqual(reopened.get_run(self.account_id, run.run_id).account_snapshot, self.account)
        events = reopened.list_events(self.account_id, run.run_id)
        self.assertEqual([event.seq for event in events], list(range(1, len(events) + 1)))
        self.assertEqual(reopened.list_events(self.account_id, run.run_id, after_seq=2, limit=2), events[2:4])

    def test_interruption_requires_explicit_call_and_does_not_cross_owner(self):
        run = self.new_run()
        self.repository.start_run(self.account_id, run.run_id)
        self.complete_first(run)
        other = MediaRepository(self.path, owner=OwnerScope(user_id="other-user"))
        self.assertEqual(other.mark_interrupted_runs(), 0)
        reopened = MediaRepository(self.path)
        self.assertEqual(reopened.mark_interrupted_runs(), 1)
        self.assertEqual(reopened.mark_interrupted_runs(), 0)
        self.assertEqual(reopened.get_run(self.account_id, run.run_id).error.code, "INTERRUPTED")
        self.assertEqual(reopened.list_tasks(self.account_id, run.run_id)[0].status, TaskStatus.COMPLETED)

    def test_result_and_event_write_roll_back_together(self):
        run = self.new_run()
        self.repository.start_run(self.account_id, run.run_id)
        task = self.repository.list_tasks(self.account_id, run.run_id)[0]
        self.repository.start_task(self.account_id, run.run_id, task.task_id)
        before = self.repository.list_events(self.account_id, run.run_id)
        with patch.object(self.repository, "_event", side_effect=RuntimeError("模拟事件写入失败")):
            with self.assertRaises(RuntimeError):
                self.repository.complete_task(self.account_id, run.run_id, task.task_id, self.result())
        restored = self.repository.get_task(self.account_id, run.run_id, task.task_id)
        self.assertEqual(restored.status, TaskStatus.RUNNING)
        self.assertIsNone(restored.result)
        self.assertEqual(self.repository.list_events(self.account_id, run.run_id), before)

    def test_models_are_revalidated_and_corrupt_database_payload_is_rejected(self):
        invalid = account_config().model_copy(update={"publishing_frequency": True})
        with self.assertRaises(ValidationError):
            self.repository.create_account(invalid)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("UPDATE media_account SET snapshot_json='{}' WHERE account_id=?", (self.account_id,))
        with self.assertRaises(ValidationError):
            self.repository.get_account(self.account_id)

    def test_run_creation_and_initial_event_are_atomic(self):
        with patch.object(self.repository, "_event", side_effect=RuntimeError("模拟提交失败")):
            with self.assertRaises(RuntimeError):
                self.new_run()
        self.assertEqual(self.repository.list_runs(self.account_id), [])
        run = self.new_run()
        self.assertEqual(len(self.repository.list_tasks(self.account_id, run.run_id)), 6)
        self.assertEqual(len(self.repository.list_events(self.account_id, run.run_id)), 1)

    def test_foreign_keys_reject_cross_run_task_dependencies(self):
        first_run, second_run = self.new_run(), self.new_run("second")
        first_task = self.repository.list_tasks(self.account_id, first_run.run_id)[0]
        second_task = self.repository.list_tasks(self.account_id, second_run.run_id)[0]
        with self.assertRaises(sqlite3.IntegrityError):
            with self.repository._connection(write=True) as connection:
                connection.execute("INSERT INTO task_dependency VALUES (?, ?, ?, ?)",
                                   (first_run.run_id, first_task.task_id, second_task.task_id, 0))

    def test_concurrent_account_updates_do_not_lose_changes(self):
        def update(tone):
            try:
                return self.repository.update_account(self.account_id, account_config(tone=tone), expected_version=1)
            except ConflictError:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            accounts = list(pool.map(update, ["风格 A", "风格 B"]))
        self.assertEqual(sum(account is not None for account in accounts), 1)
        self.assertEqual(len(self.repository.account_history(self.account_id)), 2)

    def test_waiting_approval_can_be_cancelled_without_deleting_results(self):
        run = self.new_run()
        self.repository.start_run(self.account_id, run.run_id)
        for task in self.repository.list_tasks(self.account_id, run.run_id):
            self.repository.start_task(self.account_id, run.run_id, task.task_id)
            self.repository.complete_task(self.account_id, run.run_id, task.task_id, self.result())
        self.repository.finish_run(self.account_id, run.run_id)
        self.repository.cancel_run(self.account_id, run.run_id)
        self.assertTrue(all(task.status == TaskStatus.COMPLETED
                            for task in self.repository.list_tasks(self.account_id, run.run_id)))


if __name__ == "__main__":
    unittest.main()
