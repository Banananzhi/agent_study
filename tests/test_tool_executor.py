import unittest

from tool_executor import ToolExecutor
from tool_result import ErrorCode
from tools import RetryPolicy, Tool, ToolAuthenticationError


SCHEMA = {
    "type": "function",
    "function": {
        "name": "fake",
        "description": "测试工具",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}


# 创建带重试策略的测试工具
# function：测试工具函数
# max_attempts：最大执行次数
# idempotent：是否允许重复执行
def make_tool(function, max_attempts=3, idempotent=True):
    return Tool(
        function=function,
        schema=SCHEMA,
        display_name="Fake",
        retry_policy=RetryPolicy(
            max_attempts=max_attempts,
            base_delay=0.5,
            max_delay=2.0,
            jitter=0.2,
        ),
        idempotent=idempotent,
    )


class ToolExecutorTests(unittest.TestCase):
    # 验证临时超时后能够自动重试成功
    def test_retry_until_success(self):
        calls = []
        delays = []

        # 前两次调用超时，第三次返回成功
        def flaky_tool():
            calls.append(1)
            if len(calls) < 3:
                raise TimeoutError
            return "ok"

        executor = ToolExecutor(
            {"fake": make_tool(flaky_tool)},
            sleeper=delays.append,
            jitter_fn=lambda start, end: 0,
        )
        result = executor.execute("fake", {})

        self.assertTrue(result.ok)
        self.assertEqual(result.value, "ok")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(delays, [0.5, 1.0])

    # 验证重试耗尽后返回最终失败
    def test_retry_exhausted(self):
        calls = []
        delays = []

        # 每次调用都模拟超时
        def timeout_tool():
            calls.append(1)
            raise TimeoutError

        executor = ToolExecutor(
            {"fake": make_tool(timeout_tool)},
            sleeper=delays.append,
            jitter_fn=lambda start, end: 0,
        )
        result = executor.execute("fake", {})

        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ErrorCode.TIMEOUT)
        self.assertEqual(result.attempts, 3)
        self.assertTrue(result.retry_exhausted)
        self.assertEqual(len(calls), 3)
        self.assertEqual(delays, [0.5, 1.0])

    # 验证认证错误不会自动重试
    def test_authentication_error_does_not_retry(self):
        calls = []

        # 模拟工具服务认证失败
        def auth_error_tool():
            calls.append(1)
            raise ToolAuthenticationError("认证失败")

        executor = ToolExecutor(
            {"fake": make_tool(auth_error_tool)},
            sleeper=lambda delay: self.fail("认证错误不应等待重试"),
        )
        result = executor.execute("fake", {})

        self.assertEqual(result.error_code, ErrorCode.AUTHENTICATION_ERROR)
        self.assertEqual(result.attempts, 1)
        self.assertFalse(result.retry_exhausted)
        self.assertEqual(len(calls), 1)

    # 验证非幂等工具即使超时也不会自动重试
    def test_non_idempotent_tool_does_not_retry(self):
        calls = []

        # 模拟具有副作用的工具超时
        def side_effect_tool():
            calls.append(1)
            raise TimeoutError

        executor = ToolExecutor(
            {"fake": make_tool(side_effect_tool, idempotent=False)},
            sleeper=lambda delay: self.fail("非幂等工具不应等待重试"),
        )
        result = executor.execute("fake", {})

        self.assertEqual(result.error_code, ErrorCode.TIMEOUT)
        self.assertEqual(result.attempts, 1)
        self.assertTrue(result.retry_exhausted)
        self.assertEqual(len(calls), 1)

    # 验证参数校验失败时不会调用工具或进入重试
    def test_invalid_arguments_do_not_execute(self):
        called = []
        schema = {
            "type": "function",
            "function": {
                "name": "fake",
                "description": "测试工具",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
        tool = Tool(lambda value: called.append(value), schema, "Fake")
        result = ToolExecutor({"fake": tool}).execute("fake", {})

        self.assertEqual(result.error_code, ErrorCode.INVALID_ARGUMENTS)
        self.assertEqual(result.attempts, 0)
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
