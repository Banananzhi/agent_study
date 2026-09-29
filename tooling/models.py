from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, RootModel, StrictFloat, StrictInt


# FiniteFloat：不允许 NaN 和无穷大的严格浮点数类型
FiniteFloat = Annotated[StrictFloat, Field(allow_inf_nan=False)]


class CalculatorOutput(RootModel):
    """返回有限的整数或浮点数计算结果。"""

    model_config = ConfigDict(strict=True)
    root: StrictInt | FiniteFloat


class WebSearchOutput(RootModel):
    """返回格式化搜索文本，每条包含序号、标题、URL 和摘要。"""

    model_config = ConfigDict(strict=True)
    root: str = Field(min_length=1)


class ReadWebpageOutput(RootModel):
    """返回网页的可见正文文本，不包含 script 和 style 内容。"""

    model_config = ConfigDict(strict=True)
    root: str = Field(min_length=1, max_length=50000)


class CurrentTimeOutput(RootModel):
    """返回带时区偏移的 ISO 8601 日期时间。"""

    model_config = ConfigDict(strict=True)
    root: AwareDatetime


class FileOperationOutput(BaseModel):
    """返回文件操作后的工作区相对路径、写入字符数和下一写入偏移。"""

    model_config = ConfigDict(strict=True, extra="forbid")
    path: str = Field(min_length=1, description="操作文件的工作区相对路径")
    chars_written: StrictInt = Field(ge=0, description="成功写入的字符数")
    next_offset: StrictInt = Field(ge=0, description="下一段内容应使用的字符偏移")


class ReadFileOutput(BaseModel):
    """返回文件的分页位置、UTF-8 文本内容和是否还有后续内容。"""

    model_config = ConfigDict(strict=True, extra="forbid")
    path: str = Field(min_length=1, description="读取文件的工作区相对路径")
    offset: StrictInt = Field(ge=0, description="本次读取的起始字符偏移")
    next_offset: StrictInt = Field(ge=0, description="下一页应使用的字符偏移")
    has_more: bool = Field(description="文件在本页之后是否还有内容")
    content: str = Field(description="从文件中读取的 UTF-8 文本内容")
