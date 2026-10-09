import unittest
from datetime import UTC, datetime, timedelta, timezone

from pydantic import ValidationError

from media_operations.schemas import (
    AccountBrief, AccountCreate, AgentTaskResult, GoalDefinition, Payload,
    RunCreate, StrategyCreate, TaskSpec, TaskType,
)


def account_config(**changes):
    return AccountCreate(**{
        "account_name": "AI 技术分享", "positioning": "AI Agent 开发实战",
        "target_audience": "Java 开发者", "tone": "准确、清晰",
        "content_pillars": ["教程", "实战"], **changes,
    })


class MediaSchemaTests(unittest.TestCase):
    def test_rejects_invalid_account_and_untrusted_identity_fields(self):
        for changes in ({"account_name": "  "}, {"publishing_frequency": True},
                        {"content_pillars": []}, {"content_pillars": ["教程", " 教程 "]},
                        {"tenant_id": "other"}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                account_config(**changes)

    def test_strategy_goals_weights_and_finite_values(self):
        for weights in ({"教程": -0.1}, {"教程": 0.4}, {"教程": float("nan")}, {"教程": float("inf")}):
            with self.subTest(weights=weights), self.assertRaises(ValidationError):
                StrategyCreate(pillar_weights=weights)
        with self.assertRaises(ValidationError):
            GoalDefinition(metric="篇数", period_start="2026-10-10", period_end="2026-10-01", rationale="目标")

    def test_plan_rejects_duplicate_missing_forward_and_cyclic_dependencies(self):
        first = TaskSpec(task_key="a", task_type=TaskType.RESEARCH)
        cases = [
            [first, first],
            [TaskSpec(task_key="a", task_type=TaskType.RESEARCH, depends_on=["missing"])],
            [TaskSpec(task_key="a", task_type=TaskType.RESEARCH, depends_on=["b"]),
             TaskSpec(task_key="b", task_type=TaskType.CONTENT, depends_on=["a"])],
            [first, TaskSpec(task_key="b", task_type=TaskType.CONTENT, depends_on=["a", "a"])],
        ]
        for tasks in cases:
            with self.subTest(tasks=tasks), self.assertRaises(ValidationError):
                RunCreate(idempotency_key="key", goal="任务", tasks=tasks)

    def test_results_require_consistent_success_or_error(self):
        for result in ({"success": True}, {"success": False, "data": {}},
                       {"success": True, "data": {}, "error": {"code": "ERR", "message": "失败"}}):
            with self.subTest(result=result), self.assertRaises(ValidationError):
                AgentTaskResult[Payload].model_validate(result)
        with self.assertRaises(ValidationError):
            AgentTaskResult[Payload](success=True, data={"not_json": object()})
        for data in ({"nested": [float("nan")]}, {"nested": {"number": float("inf")}}):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                AgentTaskResult[Payload](success=True, data=data)
        with self.assertRaises(ValidationError):
            AgentTaskResult[Payload](success="true", data={})

    def test_timestamps_are_aware_and_normalized_to_utc(self):
        now = datetime(2026, 10, 9, 20, tzinfo=timezone(timedelta(hours=8)))
        account = AccountBrief(**account_config().model_dump(), account_id="a", version=1, created_at=now, updated_at=now)
        self.assertEqual(account.created_at, datetime(2026, 10, 9, 12, tzinfo=UTC))
        self.assertEqual(account.created_at.utcoffset(), timedelta(0))
        with self.assertRaises(ValidationError):
            AccountBrief(**account_config().model_dump(), account_id="a", version=1,
                         created_at=now.replace(tzinfo=None), updated_at=now)


if __name__ == "__main__":
    unittest.main()
