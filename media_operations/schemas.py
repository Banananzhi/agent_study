"""阶段 1A 的领域契约；不依赖 LangGraph、工具消息或供应商 SDK。"""

from datetime import UTC, date, datetime
from enum import Enum
from math import isfinite
from typing import Annotated, Generic, TypeVar

from pydantic import (
    AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, StrictBool, StrictInt,
    StringConstraints, field_validator, model_validator,
)


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
Payload = dict[str, JsonValue]
T = TypeVar("T")


def _check_finite(value):
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("JSON 数值不能是 NaN 或无穷大")
    if isinstance(value, dict):
        for item in value.values():
            _check_finite(item)
    elif isinstance(value, list):
        for item in value:
            _check_finite(item)


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @field_validator("*", mode="after")
    @classmethod
    def utc_datetimes(cls, value):
        if isinstance(value, datetime):
            return value.astimezone(UTC)
        _check_finite(value)
        return value


class OwnerScope(DomainModel):
    """由可信入口注入；账号输入和模型输出不能指定归属。"""

    tenant_id: Identifier = "local"
    user_id: Identifier = "local-user"


class AccountStatus(str, Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"


class AccountCreate(DomainModel):
    platform: Identifier = "xiaohongshu"
    account_name: Identifier
    positioning: Text
    target_audience: Text
    tone: Text
    content_pillars: list[Identifier] = Field(min_length=1, max_length=30)
    publishing_frequency: StrictInt = Field(default=3, ge=1, le=100, description="每周建议篇数")
    banned_terms: list[Identifier] = Field(default_factory=list, max_length=200)
    status: AccountStatus = AccountStatus.ACTIVE

    @field_validator("content_pillars", "banned_terms")
    @classmethod
    def unique_terms(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("列表不能包含重复项")
        return value


class AccountBrief(AccountCreate):
    account_id: Identifier
    version: StrictInt = Field(ge=1)
    created_at: AwareDatetime
    updated_at: AwareDatetime


class GoalDefinition(DomainModel):
    metric: Identifier
    target_value: float | None = Field(default=None, ge=0)
    period_start: date
    period_end: date
    rationale: Text

    @model_validator(mode="after")
    def valid_period(self):
        if self.period_end < self.period_start:
            raise ValueError("目标结束日期不能早于开始日期")
        return self


class StrategyCreate(DomainModel):
    pillar_weights: dict[Identifier, float] = Field(default_factory=dict)
    quality_rules: list[Text] = Field(default_factory=list, max_length=50)
    goals: list[GoalDefinition] = Field(default_factory=list, max_length=30)
    rationale: Text = "用户配置的初始内容策略"

    @field_validator("pillar_weights")
    @classmethod
    def valid_weights(cls, value):
        if any(weight < 0 or weight > 1 for weight in value.values()):
            raise ValueError("内容占比必须在 0–1 之间")
        if value and abs(sum(value.values()) - 1) > 1e-6:
            raise ValueError("内容占比之和必须为 1")
        return value


class ContentStrategy(StrategyCreate):
    strategy_id: Identifier
    account_id: Identifier
    version: StrictInt = Field(ge=1)
    created_at: AwareDatetime


class RunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class TaskType(str, Enum):
    COORDINATOR = "coordinator"
    RESEARCH = "research"
    PLANNING = "planning"
    CONTENT = "content"
    REVIEW = "review"
    PERSIST_EXPORT = "persist_export"


class RunBudget(DomainModel):
    """当前仅保存预算；模型/工具使用量计量和执行限制在 1C 接入。"""

    max_model_calls: StrictInt = Field(default=24, ge=1)
    max_tool_calls: StrictInt = Field(default=30, ge=1)
    max_tokens: StrictInt = Field(default=120000, ge=1)
    max_duration_seconds: StrictInt = Field(default=1200, ge=1)
    max_content_revisions: StrictInt = Field(default=2, ge=0, le=10)


class TaskSpec(DomainModel):
    task_key: Identifier
    task_type: TaskType
    depends_on: list[Identifier] = Field(default_factory=list)
    input: Payload = Field(default_factory=dict)


def default_tasks():
    """固定 MVP 模板；阶段 1A 只保存计划，不执行内容生产。"""
    tasks = []
    for task_type in TaskType:
        tasks.append(TaskSpec(
            task_key=task_type.value, task_type=task_type,
            depends_on=[tasks[-1].task_key] if tasks else [],
        ))
    return tasks


class RunCreate(DomainModel):
    idempotency_key: Identifier
    goal: Text
    budget: RunBudget = Field(default_factory=RunBudget)
    tasks: list[TaskSpec] = Field(default_factory=default_tasks, min_length=1, max_length=100)

    @model_validator(mode="after")
    def ordered_dependencies(self):
        # 使用按依赖排序的计划；禁止跨 Run 引用、重复、前向依赖和循环。
        seen = set()
        for task in self.tasks:
            if task.task_key in seen:
                raise ValueError("任务 key 不能重复")
            if len(task.depends_on) != len(set(task.depends_on)):
                raise ValueError("任务依赖不能重复")
            if not set(task.depends_on) <= seen:
                raise ValueError("任务依赖必须引用计划中已定义的前置任务")
            seen.add(task.task_key)
        return self


class TaskError(DomainModel):
    code: Identifier
    message: Text
    retryable: StrictBool = False


class AgentTaskResult(DomainModel, Generic[T]):
    success: StrictBool
    data: T | None = None
    error: TaskError | None = None
    source_ids: list[Identifier] = Field(default_factory=list)
    artifact_refs: list[Identifier] = Field(default_factory=list)
    warnings: list[Text] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_result(self):
        if self.success and (self.data is None or self.error is not None):
            raise ValueError("成功结果必须包含 data 且不能包含 error")
        if not self.success and (self.data is not None or self.error is None):
            raise ValueError("失败结果必须包含 error 且不能包含 data")
        return self


class AgentRun(DomainModel):
    run_id: Identifier
    account_id: Identifier
    request: RunCreate
    account_snapshot: AccountBrief
    strategy_snapshot: ContentStrategy
    status: RunStatus
    error: TaskError | None = None
    version: StrictInt = Field(ge=1)
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def snapshots_belong_to_account(self):
        if self.account_snapshot.account_id != self.account_id or self.strategy_snapshot.account_id != self.account_id:
            raise ValueError("Run 快照必须属于当前账号")
        if (self.status in {RunStatus.FAILED, RunStatus.CANCELLED}) != (self.error is not None):
            raise ValueError("失败/取消 Run 必须包含错误，其他状态不能包含错误")
        return self


class AgentTask(DomainModel):
    task_id: Identifier
    run_id: Identifier
    task_key: Identifier
    task_type: TaskType
    dependencies: list[Identifier]
    input: Payload
    status: TaskStatus
    attempts: StrictInt = Field(ge=0)
    result: AgentTaskResult[Payload] | None = None
    skip_reason: Text | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def consistent_status(self):
        if self.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
            if self.result is None or self.result.success != (self.status == TaskStatus.COMPLETED):
                raise ValueError("Task 状态与结果不一致")
        elif self.result is not None:
            raise ValueError("未完成 Task 不能包含结果")
        if (self.status == TaskStatus.SKIPPED) != (self.skip_reason is not None):
            raise ValueError("只有 SKIPPED Task 必须包含跳过原因")
        return self


class RunEvent(DomainModel):
    event_id: Identifier
    run_id: Identifier
    seq: StrictInt = Field(ge=1)
    event_type: Identifier
    payload: Payload
    created_at: AwareDatetime

