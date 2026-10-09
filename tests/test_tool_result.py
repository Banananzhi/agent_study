import json
import unittest

from tooling.result import ErrorCode, ToolResult


class ToolResultObservationTests(unittest.TestCase):
    # 验证未超限的 Observation 保留完整工具结果
    def test_short_observation_is_not_truncated(self):
        # result：内容足够短的工具成功结果
        result = ToolResult.success("calculator", 3)

        # observation：在限制范围内的完整观察结果
        observation = result.to_observation(512)

        # payload：从 Observation 解析出的 JSON 对象
        payload = json.loads(observation)
        self.assertEqual(payload["value"], 3)
        self.assertNotIn("truncation", payload)

    # 验证长文本结果会在保持 JSON 合法的前提下被截断
    def test_long_text_value_is_truncated(self):
        # result：包含超长文本的工具成功结果
        result = ToolResult.success("read_webpage", "网页内容" * 3000)

        # observation：长度不超过 800 字符的截断观察结果
        observation = result.to_observation(800)

        # payload：从截断 Observation 解析出的 JSON 对象
        payload = json.loads(observation)
        self.assertLessEqual(len(observation), 800)
        self.assertTrue(payload["truncation"]["truncated"])
        self.assertEqual(payload["truncation"]["field"], "value")
        self.assertEqual(payload["truncation"]["format"], "text")
        self.assertEqual(payload["truncation"]["original_field_chars"], 12000)
        self.assertTrue(payload["value"].endswith("…"))

    # 验证非字符串工具结果超限时会转换为 JSON 文本再截断
    def test_structured_value_is_truncated_as_json_text(self):
        # value：模拟工具返回的大型结构化数据
        value = {"items": [{"title": "搜索结果" * 100}] * 20}

        # result：包含大型结构化数据的成功结果
        result = ToolResult.success("web_search", value)

        # observation：结构化 value 转为文本后的截断观察结果
        observation = result.to_observation(800)

        # payload：从截断 Observation 解析出的 JSON 对象
        payload = json.loads(observation)
        self.assertLessEqual(len(observation), 800)
        self.assertIsInstance(payload["value"], str)
        self.assertEqual(payload["truncation"]["format"], "json_text")

    # 验证长错误说明被截断后仍保留错误码和恢复策略
    def test_long_error_message_is_truncated(self):
        # result：包含超长错误说明的工具失败结果
        result = ToolResult.failure(
            "calculator",
            ErrorCode.INVALID_ARGUMENTS,
            "参数错误" * 3000,
        )

        # observation：错误说明被安全截断后的观察结果
        observation = result.to_observation(800)

        # payload：从截断 Observation 解析出的 JSON 对象
        payload = json.loads(observation)
        self.assertLessEqual(len(observation), 800)
        self.assertEqual(payload["error"]["code"], "invalid_arguments")
        self.assertTrue(payload["error"]["model_recoverable"])
        self.assertEqual(payload["truncation"]["field"], "message")

    # 验证过小的 Observation 长度配置会被拒绝
    def test_observation_limit_has_safe_minimum(self):
        # result：用于校验长度配置的简短成功结果
        result = ToolResult.success("calculator", 3)

        with self.assertRaisesRegex(ValueError, "512"):
            result.to_observation(511)


if __name__ == "__main__":
    unittest.main()
