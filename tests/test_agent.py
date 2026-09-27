import copy
import io
import json
import unittest
from unittest.mock import patch

from agent import Agent, AgentToolError
from tool_result import ErrorCode, ToolResult
from tools import TOOLS


class FakeExecutor:
    # 初始化按顺序返回结果的测试执行器
    # results：每次 execute 调用应返回的结果
    # registry：用于生成 Action 指纹的测试工具注册表
    def __init__(self, results, registry=None):
        self.results = iter(results)
        self.registry = {} if registry is None else registry
        self.calls = []

    # 返回下一个预置工具结果
    # name：模型选择的工具名称
    # args：模型生成的工具参数
    def execute(self, name, args):
        self.calls.append((name, args))
        return next(self.results)


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
    # 验证 think 会通过 API 原生 tools 字段发送工具 Schema
    def test_think_uses_native_function_calling_request(self):
        # response_body：模拟 DeepSeek 返回的最终 assistant 响应
        response_body = {
            "choices": [{"message": final_response("直接答案")}],
        }

        # response_bytes：可供 json.load 读取的模拟 HTTP 响应字节
        response_bytes = json.dumps(response_body).encode("utf-8")

        # agent：用于验证真实 think 请求体的 Agent
        agent = Agent()
        agent.api_key = "test-key"

        with patch("agent.urllib.request.urlopen", return_value=io.BytesIO(response_bytes)) as urlopen:
            agent.think([{"role": "user", "content": "你好"}])

        # request：think 向 DeepSeek 接口构造的 HTTP 请求
        request = urlopen.call_args.args[0]

        # request_body：从 HTTP 请求中解析出的 Function Calling 请求体
        request_body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request_body["tool_choice"], "auto")
        self.assertEqual(len(request_body["tools"]), 4)
        self.assertNotIn("response_format", request_body)
        self.assertTrue(all("返回值：" in item["function"]["description"] for item in request_body["tools"]))

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
        observation = json.loads(tool_message["content"])
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "call_1")
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

    # 验证原生 arguments 不是合法 JSON 时会交给模型修正
    def test_invalid_native_arguments_are_recoverable(self):
        # invalid_response：arguments 字段不是合法 JSON 的原生响应
        invalid_response = tool_response("calculator")
        invalid_response["tool_calls"][0]["function"]["arguments"] = "{"
        agent = ScriptedAgent(
            [invalid_response, final_response("已终止")],
            tool_executor=FakeExecutor([]),
        )

        self.assertEqual(agent.run("计算"), "已终止")

        # observation：模型第二轮收到的参数解析错误
        observation = json.loads(agent.seen_messages[1][-1]["content"])
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
        self.assertEqual([item["role"] for item in tool_messages], ["tool", "tool"])
        self.assertEqual(
            [item["tool_call_id"] for item in tool_messages],
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
        observation = json.loads(agent.seen_messages[2][-1]["content"])
        self.assertEqual(observation["error"]["code"], "repeated_action")
        self.assertTrue(observation["error"]["model_recoverable"])
        self.assertEqual(observation["attempts"], 0)

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
    def test_agent_limits_tool_message_length(self):
        # executor：返回超长网页内容的测试执行器
        executor = FakeExecutor([
            ToolResult.success("read_webpage", "网页内容" * 3000),
        ])
        agent = ScriptedAgent(
            [
                tool_response("read_webpage", {"url": "https://example.com"}, "call_1"),
                final_response("完成"),
            ],
            tool_executor=executor,
            max_observation_chars=800,
        )

        self.assertEqual(agent.run("读取网页"), "完成")

        # tool_message：第二轮模型调用前收到的受限工具消息
        tool_message = agent.seen_messages[1][-1]
        self.assertLessEqual(len(tool_message["content"]), 800)
        self.assertTrue(json.loads(tool_message["content"])["truncation"]["truncated"])


if __name__ == "__main__":
    unittest.main()
