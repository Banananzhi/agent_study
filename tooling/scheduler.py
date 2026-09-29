import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass

from tooling.resources import resource_sets_conflict
from tooling.result import ToolResult


logger = logging.getLogger(__name__)


@dataclass
class BatchToolCall:
    index: int
    tool_call_id: str
    tool_name: str
    arguments: dict | None
    action_signature: tuple | None = None
    preset_result: ToolResult | None = None
    prepared: object | None = None
    resources: tuple = ()

    # 校验批量调用的原始下标、协议标识和预置结果
    def __post_init__(self):
        if type(self.index) is not int or self.index < 0:
            raise ValueError("BatchToolCall.index 必须是非负整数")
        if not isinstance(self.tool_call_id, str) or not self.tool_call_id.strip():
            raise ValueError("BatchToolCall.tool_call_id 不能为空")
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("BatchToolCall.tool_name 不能为空")
        if self.preset_result is not None and not isinstance(self.preset_result, ToolResult):
            raise TypeError("BatchToolCall.preset_result 必须为空或 ToolResult")


class ToolBatchExecutor:
    # 初始化基于资源租约和线程池的批量工具调度器
    # tool_executor：负责工具校验、执行、重试和输出校验的执行器
    # max_parallel_tools：单批工具调用允许的最大工作线程数
    def __init__(self, tool_executor, max_parallel_tools=4):
        if type(max_parallel_tools) is not int or max_parallel_tools < 1:
            raise ValueError("max_parallel_tools 必须是大于等于 1 的整数")
        self.tool_executor = tool_executor
        self.lock_manager = tool_executor.lock_manager
        self.max_parallel_tools = max_parallel_tools

    # 准备、动态调度并按原始顺序返回一批工具结果
    # calls：按 assistant.tool_calls 原始顺序构建的调用列表
    def execute_batch(self, calls):
        # indexes：用于确保结果列表可以按原始下标安全回填的调用下标
        indexes = [call.index for call in calls]
        if sorted(indexes) != list(range(len(calls))):
            raise ValueError("BatchToolCall.index 必须从 0 开始连续且不重复")
        # results：使用调用原始下标回填的最终结果列表
        results = [None] * len(calls)
        # waiting_calls：已通过预处理但尚未获得资源的调用
        waiting_calls = []

        # call：当前正在预处理的批量工具调用
        for call in calls:
            if call.preset_result is not None:
                results[call.index] = call.preset_result
                continue
            # prepared：经过工具查找、参数校验和资源解析的执行对象
            prepared = self.tool_executor.prepare(call.tool_name, call.arguments)
            if isinstance(prepared, ToolResult):
                results[call.index] = prepared
                continue
            call.prepared = prepared
            call.resources = prepared.resources
            waiting_calls.append(call)

        if not waiting_calls:
            return results

        logger.info(
            "⚡ 批量调度 %d 个工具调用，最大并行数 %d",
            len(waiting_calls),
            self.max_parallel_tools,
        )
        # running_futures：已占用资源并进入线程池的任务映射
        running_futures = {}
        with ThreadPoolExecutor(max_workers=self.max_parallel_tools) as pool:
            while waiting_calls or running_futures:
                # available_slots：当前线程池还可接收的任务数
                available_slots = self.max_parallel_tools - len(running_futures)
                # submitted：本轮是否有新调用成功获得资源并进入线程池
                submitted = self._submit_runnable_calls(
                    waiting_calls,
                    running_futures,
                    pool,
                    available_slots,
                )

                if running_futures:
                    # completed_futures：当前至少一个已完成的工具任务
                    completed_futures, _ = wait(
                        running_futures,
                        return_when=FIRST_COMPLETED,
                    )
                    # future：当前正在回收结果的已完成任务
                    for future in completed_futures:
                        # call：已完成任务对应的原始工具调用
                        call = running_futures.pop(future)
                        # result：线程池中的工具执行结果
                        result = future.result()
                        if not isinstance(result, ToolResult):
                            raise TypeError(
                                "ToolExecutor 必须返回 ToolResult，"
                                f"实际返回 {type(result).__name__}"
                            )
                        results[call.index] = result
                        logger.info("✅ 并行工具执行完成: %s", call.tool_name)
                elif waiting_calls and not submitted:
                    self.lock_manager.wait_for_change()

        return results

    # 将当前能一次性获得全部资源的调用提交线程池
    # waiting_calls：保持模型原始顺序的等待调用列表
    # running_futures：已提交线程池的任务映射
    # pool：本批工具调用的线程池
    # available_slots：本轮最多可以提交的任务数
    def _submit_runnable_calls(
        self,
        waiting_calls,
        running_futures,
        pool,
        available_slots,
    ):
        if available_slots <= 0:
            return False
        # selected_calls：本轮已获得资源并需从等待队列移除的调用
        selected_calls = []

        # position：当前调用在原始等待队列中的位置
        # call：当前尝试调度的工具调用
        for position, call in enumerate(waiting_calls):
            if len(selected_calls) >= available_slots:
                break
            # earlier_waiting：排在当前调用之前且尚未执行的调用
            earlier_waiting = waiting_calls[:position]
            if any(
                resource_sets_conflict(call.resources, earlier.resources)
                for earlier in earlier_waiting
            ):
                continue

            # lease：在提交线程池前原子预留的全部资源
            lease = self.lock_manager.try_acquire(call.tool_call_id, call.resources)
            if lease is None:
                continue
            # future：已获得资源后才提交的工具执行任务
            try:
                future = pool.submit(
                    self.tool_executor.execute_prepared,
                    call.prepared,
                    lease,
                )
            except Exception:
                # 线程池提交本身失败时必须立即释放已预留资源
                lease.release()
                raise
            running_futures[future] = call
            selected_calls.append(call)
            logger.info("🚀 工具进入线程池: %s", call.tool_name)

        # call：本轮已提交、不再需要留在等待队列的调用
        for call in selected_calls:
            waiting_calls.remove(call)
        return bool(selected_calls)
