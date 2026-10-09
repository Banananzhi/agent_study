import copy
import io
import json
import os
import threading
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from agent import Agent, AgentToolError
from agent.summarizer import ResultSummarizer
from tooling.registry import TOOLS, format_tool_action
from tooling.resources import ResourceLockManager
from tooling.result import ErrorCode, ToolResult


class FakeExecutor:
    # 初始化按顺序返回结果的测试执行器
    # results：每次 execute 调用应返回的结果
    # registry：用于生成 Action 指纹的测试工具注册表
    def __init__(self, results, registry=None):
        self.results = iter(results)
        self.registry = {} if registry is None else registry
        self.calls = []
        self.lock_manager = ResourceLockManager()
        self.result_lock = threading.Lock()

    # 为批量调度器准备带资源声明的测试调用
    # name：测试工具名称
    # args：测试工具参数
    def prepare(self, name, args):
        # tool：测试注册表中可选的真实工具定义
        tool = self.registry.get(name)
        # resources：真实工具定义能够提供的本次资源声明
        resources = tool.resolve_resources(args) if tool is not None else ()
        return SimpleNamespace(name=name, args=args, resources=resources)

    # 在调度器预留的资源租约中返回下一个测试结果
    # prepared：测试 prepare 生成的调用字典
    # resource_lease：批量调度器预留的资源租约
    def execute_prepared(self, prepared, resource_lease):
        with resource_lease:
            with self.result_lock:
                self.calls.append((prepared.name, prepared.args))
                return next(self.results)

    # 返回下一个预置工具结果
    # name：模型选择的工具名称
    # args：模型生成的工具参数
    def execute(self, name, args):
        self.calls.append((name, args))
        return next(self.results)


class FakeSummarizer:
    # 初始化可记录调用的测试摘要器
    # summary：每次摘要调用返回的固定文本
    # error：需要模拟的摘要异常
    def __init__(self, summary="工具结果摘要", error=None):
        self.summary = summary
        self.error = error
        self.calls = []

    # 记录完整工具数据并返回预置摘要
    # tool_name：待摘要的工具名称
    # value：待摘要的完整工具业务数据
    # goal：用户当前任务目标
    # max_chars：摘要的目标最大字符数
    def summarize(self, tool_name, value, goal, max_chars):
        self.calls.append((tool_name, value, goal, max_chars))
        if self.error:
            raise self.error
        return self.summary


class ScriptedAgent(Agent):
    # 初始化使用固定原生响应的测试 Agent
    # responses：模型应按顺序返回的 assistant 消息
    # attrs：传递给 Agent 的其他配置
    def __init__(self, responses, **attrs):
        super().__init__(**attrs)
        self.responses = iter(responses)
        self.seen_messages = []

    # 模拟模型返回并保存当时的上下文
    # messages：发送给模型的上下文消息
    def think(self, messages):
        self.seen_messages.append(copy.deepcopy(messages))
        return next(self.responses)


# 创建包含一个原生工具调用的 assistant 消息
# name：模型选择的工具名称
# args：模型生成的工具参数
# call_id：原生工具调用标识
def tool_response(name="missing_tool", args=None, call_id="call_1"):
    # arguments：需要序列化为 JSON 字符串的工具参数
    arguments = {} if args is None else args
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }],
    }


# 创建不再调用工具的最终 assistant 消息
# answer：模型生成的最终答案
def final_response(answer):
    return {"role": "assistant", "content": answer}


class AgentFunctionCallingTests(unittest.TestCase):
    # 验证 Action 日志按 Schema 顺序显示查询文本而不是首个 JSON 参数
    def test_action_log_uses_schema_argument_order(self):
        # action：模型将 count 放在 query 前面时生成的搜索 Action 日志
        action = format_tool_action(
            "web_search",
            {"count": 5, "query": "华为 Pura 90 官方起售价"},
            TOOLS,
        )

        self.assertEqual(
            action,
            "WebSearch[query=华为 Pura 90 官方起售价, count=5]",
        )

    # 验证写文件 Action 日志不会输出完整文件内容
    def test_action_log_hides_large_content(self):
        # action：包含较长文件正文的写入 Action 日志
        action = format_tool_action(
            "write_file",
            {"content": "敏感正文" * 100, "path": "output/report.md", "offset": 0},
            TOOLS,
        )

        self.assertIn("path=output/report.md", action)
        self.assertIn("content=<400 字符>", action)
        self.assertNotIn("敏感正文", action)

    # 验证 think 通过 LangChain bind_tools 绑定当前注册表并调用模型
    def test_think_uses_langchain_tool_binding(self):
        # bound_model：模拟 bind_tools 返回的 LangChain Runnable
        bound_model = MagicMock()
        bound_model.invoke.return_value = AIMessage(content="直接答案")
        # chat_model：模拟可绑定工具的 LangChain ChatModel
        chat_model = MagicMock()
        chat_model.bind_tools.return_value = bound_model
        # agent：注入 ChatModel，避免测试发起真实网络请求
        agent = Agent(chat_model=chat_model)

        # message：LangChain 标准化后的模型响应
        message = agent.think([HumanMessage("你好")])

        # schemas：传递给 bind_tools 的 Function Calling 工具定义
        schemas = chat_model.bind_tools.call_args.args[0]
        self.assertEqual(chat_model.bind_tools.call_args.kwargs["tool_choice"], "auto")
        self.assertEqual(len(schemas), len(TOOLS))
        # tool_names：LangChain 实际绑定的工具注册名称集合
        tool_names = {item["function"]["name"] for item in schemas}
        self.assertIn("create_file", tool_names)
        self.assertIn("write_file", tool_names)
        self.assertTrue(all("返回值：" in item["function"]["description"] for item in schemas))
        bound_model.invoke.assert_called_once()
        self.assertEqual(message.content, "直接答案")

    # 验证可恢复错误会使用 tool 消息返回模型修正
    def test_model_recoverable_error_returns_as_tool_message(self):
        # failure：第一次原生工具调用的未知工具结果
        failure = ToolResult.failure(
            "missing_tool",
            ErrorCode.UNKNOWN_TOOL,
            "未知工具",
            suggestions=("calculator",),
        )

        # success：模型修正工具名称后的成功结果
        success = ToolResult.success("calculator", 3)
        agent = ScriptedAgent(
            [
                tool_response(),
                tool_response("calculator", {"expression": "1+2"}, "call_2"),
                final_response("3"),
            ],
            tool_executor=FakeExecutor([failure, success]),
        )

        self.assertEqual(agent.run("计算"), "3")

        # tool_message：第二次模型调用前收到的错误工具消息
        tool_message = agent.seen_messages[1][-1]

        # observation：错误工具消息中的结构化观察结果
        observation = json.loads(tool_message.content)
        self.assertEqual(tool_message.type, "tool")
        self.assertEqual(tool_message.tool_call_id, "call_1")
        self.assertTrue(observation["error"]["model_recoverable"])
        self.assertEqual(observation["error"]["suggestions"], ["calculator"])

    # 验证不可恢复错误会立即终止且不再调用模型
    def test_unrecoverable_error_stops_immediately(self):
        # failure：不允许交给模型修正的认证错误
        failure = ToolResult.failure(
            "web_search",
            ErrorCode.AUTHENTICATION_ERROR,
            "API Key 无效",
            attempts=1,
        )
        agent = ScriptedAgent(
            [tool_response("web_search")],
            tool_executor=FakeExecutor([failure]),
        )

        with self.assertRaisesRegex(AgentToolError, "authentication_error"):
            agent.run("搜索")
        self.assertEqual(len(agent.seen_messages), 1)

    # 验证连续参数错误超过上限后会终止纠错循环
    def test_model_recovery_limit(self):
        # failures：连续三轮返回的可恢复参数错误
        failures = [
            ToolResult.failure("calculator", ErrorCode.INVALID_ARGUMENTS, "参数错误")
            for _ in range(3)
        ]

        # responses：模型连续三轮生成的原生工具调用
        responses = [
            tool_response("calculator", call_id=f"call_{index}")
            for index in range(1, 4)
        ]
        agent = ScriptedAgent(
            responses,
            tool_executor=FakeExecutor(failures),
            max_model_recoveries=2,
        )

        with self.assertRaisesRegex(AgentToolError, "纠错次数已达上限"):
            agent.run("计算")
        self.assertEqual(len(agent.seen_messages), 3)

    # 验证模型不返回 tool_calls 时会直接结束任务
    def test_answer_without_tool_call_finishes_task(self):
        agent = ScriptedAgent(
            [final_response("直接答案")],
            tool_executor=FakeExecutor([]),
        )

        self.assertEqual(agent.run("你好"), "直接答案")
        self.assertEqual(len(agent.seen_messages), 1)

    # 验证相同 thread_id 可以在 Agent 重启后恢复上一轮会话消息
    def test_sqlite_checkpointer_restores_multi_turn_session(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            # checkpoint_path：测试专用的 SQLite 会话文件
            checkpoint_path = os.path.join(temporary_directory, "sessions.sqlite3")
            # first_agent：写入第一轮会话并模拟程序退出的 Agent
            first_agent = ScriptedAgent(
                [final_response("好的，你叫小王。")],
                tool_executor=FakeExecutor([]),
                checkpoint_path=checkpoint_path,
            )
            try:
                first_agent.run("我叫小王", thread_id="conversation-001")
            finally:
                first_agent.close()

            # second_agent：重新打开同一 SQLite 文件并继续同一会话的 Agent
            second_agent = ScriptedAgent(
                [final_response("你叫小王。")],
                tool_executor=FakeExecutor([]),
                checkpoint_path=checkpoint_path,
            )
            try:
                answer = second_agent.run(
                    "我叫什么？",
                    thread_id="conversation-001",
                )
                # restored_messages：第二轮请求发给模型的完整会话上下文
                restored_messages = second_agent.seen_messages[0]
                self.assertEqual(answer, "你叫小王。")
                self.assertEqual(
                    [message.type for message in restored_messages],
                    ["system", "human", "ai", "human"],
                )
                self.assertEqual(restored_messages[1].content, "我叫小王")
                self.assertEqual(restored_messages[2].content, "好的，你叫小王。")
                self.assertEqual(restored_messages[3].content, "我叫什么？")
            finally:
                second_agent.close()

    # 验证滚动会话摘要和游标会随 SQLite Checkpoint 跨 Agent 重启恢复
    def test_sqlite_checkpointer_restores_session_summary(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            # checkpoint_path：测试滚动摘要持久化的 SQLite 文件
            checkpoint_path = os.path.join(temporary_directory, "summary.sqlite3")
            # summarizer：在小上下文预算测试中返回固定会话摘要
            summarizer = FakeSummarizer("已持久化的会话摘要")
            # responses：用较长最终回答快速触发会话历史压缩
            responses = [
                final_response(str(index) * 1300)
                for index in range(1, 7)
            ]
            # first_agent：连续写入多轮对话并生成持久化摘要的 Agent
            first_agent = ScriptedAgent(
                responses,
                tool_executor=FakeExecutor([]),
                result_summarizer=summarizer,
                checkpoint_path=checkpoint_path,
                max_context_tokens=4096,
                context_compression_ratio=0.75,
                context_target_ratio=0.5,
                recent_turns_to_keep=6,
                minimum_recent_turns=2,
                session_summary_target_tokens=128,
                session_summary_max_tokens=256,
            )
            try:
                for turn_index in range(1, 7):
                    first_agent.run(
                        f"第 {turn_index} 轮问题",
                        thread_id="summary-session",
                    )
                # saved_state：第一个 Agent 结束前 SQLite 中的最新会话状态
                saved_state = first_agent.graph.get_state({
                    "configurable": {"thread_id": "summary-session"},
                }).values
                self.assertEqual(
                    saved_state["session_summary"],
                    "已持久化的会话摘要",
                )
                self.assertIsNotNone(saved_state["summary_cursor"])
            finally:
                first_agent.close()

            # second_agent：重新打开 SQLite 并恢复摘要视图的 Agent
            second_agent = ScriptedAgent(
                [final_response("继续对话")],
                tool_executor=FakeExecutor([]),
                result_summarizer=FakeSummarizer("不应重复生成"),
                checkpoint_path=checkpoint_path,
                max_context_tokens=4096,
                context_compression_ratio=0.75,
                context_target_ratio=0.5,
                recent_turns_to_keep=6,
                minimum_recent_turns=2,
                session_summary_target_tokens=128,
                session_summary_max_tokens=256,
            )
            try:
                second_agent.run("继续", thread_id="summary-session")
                # summary_messages：重启后本次模型输入中的持久化摘要消息
                summary_messages = [
                    message
                    for message in second_agent.seen_messages[0]
                    if isinstance(message, SystemMessage)
                    and message.additional_kwargs.get("session_summary")
                ]
                self.assertEqual(len(summary_messages), 1)
                self.assertIn("已持久化的会话摘要", summary_messages[0].content)
            finally:
                second_agent.close()

    # 验证原生 arguments 不是合法 JSON 时会交给模型修正
    def test_invalid_native_arguments_are_recoverable(self):
        # invalid_response：arguments 字段不是合法 JSON 的原生响应
        invalid_response = AIMessage(
            content="",
            invalid_tool_calls=[{
                "name": "calculator",
                "args": "{",
                "id": "call_1",
                "error": "工具参数不是合法 JSON",
                "type": "invalid_tool_call",
            }],
        )
        agent = ScriptedAgent(
            [invalid_response, final_response("已终止")],
            tool_executor=FakeExecutor([]),
        )

        self.assertEqual(agent.run("计算"), "已终止")

        # observation：模型第二轮收到的参数解析错误
        observation = json.loads(agent.seen_messages[1][-1].content)
        self.assertEqual(observation["error"]["code"], "invalid_arguments")
        self.assertTrue(observation["error"]["model_recoverable"])

    # 验证同一轮的多个 tool_calls 都会获得对应的 tool 消息
    def test_multiple_tool_calls_return_all_tool_messages(self):
        # first_call：模型本轮生成的第一个工具调用
        first_call = tool_response("calculator", {"expression": "1+2"}, "call_1")["tool_calls"][0]

        # second_call：模型本轮生成的第二个工具调用
        second_call = tool_response("get_current_time", {}, "call_2")["tool_calls"][0]

        # batch_response：同时包含两个原生工具调用的 assistant 消息
        batch_response = {
            "role": "assistant",
            "content": None,
            "tool_calls": [first_call, second_call],
        }
        agent = ScriptedAgent(
            [batch_response, final_response("完成")],
            tool_executor=FakeExecutor([
                ToolResult.success("calculator", 3),
                ToolResult.success("get_current_time", "2026-09-25T12:00:00+08:00"),
            ]),
        )

        self.assertEqual(agent.run("执行两个工具"), "完成")

        # tool_messages：第二轮模型调用前的全部工具结果消息
        tool_messages = agent.seen_messages[1][-2:]
        self.assertEqual([item.type for item in tool_messages], ["tool", "tool"])
        self.assertEqual(
            [item.tool_call_id for item in tool_messages],
            ["call_1", "call_2"],
        )

    # 验证紧邻的相同成功 Action 不会再次进入 ToolExecutor
    def test_consecutive_successful_action_is_blocked(self):
        # executor：只预置一次成功结果的测试执行器
        executor = FakeExecutor([ToolResult.success("calculator", 3)])
        agent = ScriptedAgent(
            [
                tool_response("calculator", {"expression": "1+2"}, "call_1"),
                tool_response("calculator", {"expression": "1+2"}, "call_2"),
                final_response("使用已有结果：3"),
            ],
            tool_executor=executor,
        )

        self.assertEqual(agent.run("计算 1+2"), "使用已有结果：3")
        self.assertEqual(len(executor.calls), 1)

        # observation：第三轮模型调用前收到的重复 Action 错误
        observation = json.loads(agent.seen_messages[2][-1].content)
        self.assertEqual(observation["error"]["code"], "repeated_action")
        self.assertTrue(observation["error"]["model_recoverable"])
        self.assertEqual(observation["attempts"], 0)

    # 验证上一批所有成功 Action 都会在下一轮被识别为立即重复
    def test_previous_batch_successful_actions_are_all_blocked(self):
        # executor：只为第一批两个不同 Action 提供成功结果
        executor = FakeExecutor([
            ToolResult.success("calculator", 2),
            ToolResult.success("calculator", 4),
        ])
        # first_batch：第一轮同时执行的两个不同计算 Action
        first_batch = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                tool_response("calculator", {"expression": "1+1"}, "call_1")["tool_calls"][0],
                tool_response("calculator", {"expression": "2+2"}, "call_2")["tool_calls"][0],
            ],
        }
        # repeated_batch：第二轮按相同顺序重复上一批全部 Action
        repeated_batch = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                tool_response("calculator", {"expression": "1+1"}, "call_3")["tool_calls"][0],
                tool_response("calculator", {"expression": "2+2"}, "call_4")["tool_calls"][0],
            ],
        }
        # agent：执行两批调用后使用已有结果结束任务
        agent = ScriptedAgent(
            [first_batch, repeated_batch, final_response("使用已有结果")],
            tool_executor=executor,
        )

        self.assertEqual(agent.run("分别计算 1+1 和 2+2"), "使用已有结果")
        self.assertEqual(len(executor.calls), 2)
        # observations：第三轮模型调用前收到的两个重复 Action 结果
        observations = [
            json.loads(message.content)
            for message in agent.seen_messages[2][-2:]
        ]
        self.assertEqual(
            [item["error"]["code"] for item in observations],
            ["repeated_action", "repeated_action"],
        )

    # 验证 Schema 默认值会参与 Action 指纹的标准化
    def test_action_signature_applies_schema_defaults(self):
        # executor：使用真实工具 Schema 但返回模拟搜索结果的执行器
        executor = FakeExecutor(
            [ToolResult.success("web_search", "搜索结果")],
            registry=TOOLS,
        )
        agent = ScriptedAgent(
            [
                tool_response("web_search", {"query": "华为手机"}, "call_1"),
                tool_response("web_search", {"count": 5, "query": "华为手机"}, "call_2"),
                final_response("使用首次搜索结果"),
            ],
            tool_executor=executor,
        )

        self.assertEqual(agent.run("搜索华为手机"), "使用首次搜索结果")
        self.assertEqual(len(executor.calls), 1)

    # 验证中间出现不同 Action 后，允许后续再次调用原 Action
    def test_non_consecutive_same_action_is_allowed(self):
        # executor：为三次非连续重复调用提供成功结果的执行器
        executor = FakeExecutor([
            ToolResult.success("calculator", 3),
            ToolResult.success("get_current_time", "2026-09-27T12:00:00+08:00"),
            ToolResult.success("calculator", 3),
        ])
        agent = ScriptedAgent(
            [
                tool_response("calculator", {"expression": "1+2"}, "call_1"),
                tool_response("get_current_time", {}, "call_2"),
                tool_response("calculator", {"expression": "1+2"}, "call_3"),
                final_response("完成"),
            ],
            tool_executor=executor,
        )

        self.assertEqual(agent.run("执行多步任务"), "完成")
        self.assertEqual(len(executor.calls), 3)

    # 验证 Agent 返回模型的 tool 消息不超过配置长度
    def test_agent_summarizes_long_tool_result(self):
        # executor：返回超长网页内容的测试执行器
        executor = FakeExecutor([
            ToolResult.success("read_webpage", "网页内容" * 3000),
        ], registry=TOOLS)
        # summarizer：用于验证完整工具值和用户目标的测试摘要器
        summarizer = FakeSummarizer("网页关键内容")
        agent = ScriptedAgent(
            [
                tool_response("read_webpage", {"url": "https://example.com"}, "call_1"),
                final_response("完成"),
            ],
            tool_executor=executor,
            max_observation_chars=800,
            result_summarizer=summarizer,
        )

        self.assertEqual(agent.run("读取网页"), "完成")

        # tool_message：第二轮模型调用前收到的受限工具消息
        tool_message = agent.seen_messages[1][-1]
        # observation：包含摘要文本和原始长度元数据的工具观察
        observation = json.loads(tool_message.content)
        self.assertLessEqual(len(tool_message.content), 800)
        self.assertEqual(observation["value"], "网页关键内容")
        self.assertTrue(observation["summarization"]["summarized"])
        self.assertEqual(summarizer.calls[0][0], "read_webpage")
        self.assertEqual(len(summarizer.calls[0][1]), 12000)

    # 验证摘要模型失败时 Agent 会回退到原有统一截断
    def test_summary_failure_falls_back_to_truncation(self):
        # executor：返回超长搜索结果的测试执行器
        executor = FakeExecutor(
            [ToolResult.success("web_search", "搜索结果" * 3000)],
            registry=TOOLS,
        )
        # summarizer：模拟请求超时的测试摘要器
        summarizer = FakeSummarizer(error=TimeoutError("摘要超时"))
        agent = ScriptedAgent(
            [
                tool_response("web_search", {"query": "测试"}, "call_1"),
                final_response("完成"),
            ],
            tool_executor=executor,
            max_observation_chars=800,
            result_summarizer=summarizer,
        )

        self.assertEqual(agent.run("搜索"), "完成")
        # observation：摘要失败后使用的受限工具结果
        observation = json.loads(agent.seen_messages[1][-1].content)
        self.assertTrue(observation["truncation"]["truncated"])

    # 验证公共摘要器会分块处理完整结果且摘要请求不携带工具权限
    def test_result_summarizer_uses_tool_free_chunk_requests(self):
        # response_body：每次摘要请求的模拟模型响应
        response_body = {
            "choices": [{"message": final_response("分块摘要")}],
        }

        # fake_urlopen：为每次分块和汇总请求创建独立响应流
        def fake_urlopen(request, timeout):
            return io.BytesIO(json.dumps(response_body).encode("utf-8"))

        # summarizer：使用较小分块验证 map-reduce 摘要流程的公共组件
        summarizer = ResultSummarizer(
            model="deepseek-chat",
            api_url="https://api.deepseek.com",
            api_key="test-key",
            chunk_chars=1000,
        )
        with patch("agent.summarizer.urllib.request.urlopen", side_effect=fake_urlopen) as urlopen:
            # summary：三个原始分块经过一次最终汇总后的摘要
            summary = summarizer.summarize("read_webpage", "x" * 2500, "测试目标", 500)

        self.assertEqual(summary, "分块摘要")
        self.assertEqual(urlopen.call_count, 4)
        for call in urlopen.call_args_list:
            # request_body：当前摘要 HTTP 请求的 JSON 请求体
            request_body = json.loads(call.args[0].data.decode("utf-8"))
            self.assertNotIn("tools", request_body)
            self.assertNotIn("tool_choice", request_body)


if __name__ == "__main__":
    unittest.main()
