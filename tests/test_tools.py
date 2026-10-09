import unittest
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tooling.models import CalculatorOutput, CurrentTimeOutput


class SearchItem(BaseModel):
    """表示单条搜索结果。"""

    model_config = ConfigDict(strict=True, extra="forbid")
    title: str = Field(min_length=1, description="网页标题")
    score: float = Field(ge=0, le=1, allow_inf_nan=False, description="搜索相关度")


class StructuredSearchOutput(BaseModel):
    """返回结构化搜索结果。"""

    model_config = ConfigDict(strict=True, extra="forbid")
    results: list[SearchItem]


class PydanticOutputModelTests(unittest.TestCase):
    # 验证 Pydantic 能校验并序列化符合约定的嵌套业务结果
    def test_structured_output_is_validated_and_serialized(self):
        # value：完整符合嵌套返回模型的原始业务数据
        value = {
            "results": [
                {"title": "华为官网", "score": 0.98},
                {"title": "华为商城", "score": 0.95},
            ],
        }

        # validated：经过 Pydantic 严格校验的结构化搜索结果
        validated = StructuredSearchOutput.model_validate(value)

        # serialized：转换为标准 JSON 类型的搜索结果
        serialized = validated.model_dump(mode="json")
        self.assertEqual(serialized, value)

    # 验证嵌套结果缺少必填字段时会返回精确路径
    def test_nested_required_field_is_validated(self):
        # value：第一个搜索结果缺少 score 的错误业务数据
        value = {"results": [{"title": "华为官网"}]}

        with self.assertRaises(ValidationError) as context:
            StructuredSearchOutput.model_validate(value)

        # location：Pydantic 返回的第一个嵌套错误字段路径
        location = context.exception.errors(include_url=False, include_input=False)[0]["loc"]
        self.assertEqual(location, ("results", 0, "score"))

    # 验证计算器返回模型不会将布尔值误当作数字
    def test_calculator_output_rejects_boolean(self):
        with self.assertRaises(ValidationError):
            CalculatorOutput.model_validate(True)

    # 验证计算器返回模型会拒绝无穷大
    def test_calculator_output_rejects_infinite_number(self):
        with self.assertRaises(ValidationError):
            CalculatorOutput.model_validate(float("inf"))

    # 验证当前时间返回模型要求 datetime 必须包含时区
    def test_current_time_output_requires_timezone(self):
        with self.assertRaises(ValidationError):
            CurrentTimeOutput.model_validate(datetime(2026, 9, 27, 12, 0, 0))

        # aware_datetime：可通过返回模型校验的带时区日期时间
        aware_datetime = datetime(2026, 9, 27, 12, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(
            CurrentTimeOutput.model_validate(aware_datetime).model_dump(mode="json"),
            "2026-09-27T12:00:00Z",
        )


if __name__ == "__main__":
    unittest.main()
