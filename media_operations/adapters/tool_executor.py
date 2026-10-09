"""通过公开 prepare/execute_prepared 扩展现有执行器，不重写重试与资源锁。"""

from dataclasses import dataclass, replace

from media_operations.adapters.redaction import Redactor
from tooling.errors import ClassifiedToolError
from tooling.executor import PreparedToolExecution, ToolExecutor
from tooling.result import ErrorCode, ToolResult


@dataclass(frozen=True)
class RecordedExecution(PreparedToolExecution):
    execution_id: str


class ResearchToolExecutor(ToolExecutor):
    def __init__(self, service, registry, **kwargs):
        super().__init__(registry=registry, **kwargs)
        self.service = service
        self.redactor: Redactor = service.redactor
        self.before_execute = service.check_active

    def _finish(self, execution_id, result):
        if not result.ok:
            result = replace(result, tool=self.redactor.text(result.tool),
                             error_message=self.redactor.text(result.error_message))
        self.service.repository.finish_tool_execution(
            self.service.account_id, self.service.run_id, execution_id, self.redactor.value(result.to_dict()),
        )
        return result

    def prepare(self, name, args):
        try:
            self.service.check_active()
        except ClassifiedToolError as error:
            return ToolResult.failure(self.redactor.text(name), error.error_code, str(error))
        # 存储失败则不执行网络调用，不能悄悄跳过审计。
        execution_id = self.service.repository.begin_tool_execution(
            self.service.account_id, self.service.run_id, self.service.task_id,
            self.redactor.text(name), self.redactor.value(args) if isinstance(args, dict) else {"invalid_arguments": self.redactor.value(args)},
        )
        prepared = super().prepare(name, args)
        if isinstance(prepared, ToolResult):
            return self._finish(execution_id, prepared)
        return RecordedExecution(name=prepared.name, args=prepared.args, tool=prepared.tool,
                                 resources=prepared.resources, execution_id=execution_id)

    def execute_prepared(self, prepared, resource_lease=None):
        if not isinstance(prepared, RecordedExecution):
            raise TypeError("研究执行器需要已记录的 PreparedExecution")
        result = super().execute_prepared(prepared, resource_lease)
        return self._finish(prepared.execution_id, result)
