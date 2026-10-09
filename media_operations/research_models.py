"""研究工具契约。来源是程序取得的数据，不是模型指令或事实正确性证明。"""

import hashlib
from pathlib import PureWindowsPath
from datetime import datetime
from enum import Enum
from typing import Literal
from urllib.parse import urlsplit

from pydantic import AwareDatetime, Field, StrictBool, StrictInt, field_validator, model_validator

from media_operations.schemas import DomainModel, Identifier, Payload, Text
from tooling.result import ErrorCode, ToolResult


class TimeRange(str, Enum):
    DAY = "oneDay"
    WEEK = "oneWeek"
    MONTH = "oneMonth"
    YEAR = "oneYear"


class ResearchQuery(DomainModel):
    query: str = Field(min_length=1, max_length=1000)
    time_range: TimeRange | None = None
    limit: StrictInt = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def nonempty_query(cls, value):
        if not value.strip():
            raise ValueError("查询不能为空")
        return value.strip()


class PageQuery(DomainModel):
    url: str = Field(min_length=1, max_length=2048)


class KnowledgeQuery(ResearchQuery):
    account_id: Identifier | None = None

    @model_validator(mode="after")
    def no_time_filter(self):
        if self.time_range is not None:
            raise ValueError("本地知识不支持发布时间过滤")
        return self


class SourceKind(str, Enum):
    SEARCH = "search"
    WEB = "web"
    KNOWLEDGE = "knowledge"


class SourceReference(DomainModel):
    """可回查的证据引用；retrieved/discovered 不表示事实已核实。"""

    source_id: Identifier
    kind: SourceKind
    provider: Identifier
    original_url: str | None = Field(default=None, max_length=2048)
    final_url: str | None = Field(default=None, max_length=2048)
    relative_path: str | None = Field(default=None, max_length=1000)
    title: str = Field(max_length=300)
    published_at: AwareDatetime | None = None
    published_at_raw: str | None = Field(default=None, max_length=100)
    retrieved_at: AwareDatetime
    snippet: str = Field(max_length=600)
    evidence_start: StrictInt = Field(default=0, ge=0)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    truncated: StrictBool = False
    verification_status: Literal["discovered", "retrieved"]
    untrusted_data: Literal[True] = True

    @model_validator(mode="after")
    def valid_origin(self):
        if self.kind == SourceKind.KNOWLEDGE:
            if not self.relative_path or self.original_url is not None or self.final_url is not None:
                raise ValueError("知识来源必须是受控相对路径")
            if (self.relative_path.startswith(("/", "\\")) or PureWindowsPath(self.relative_path).drive
                    or ".." in self.relative_path.replace("\\", "/").split("/")):
                raise ValueError("知识路径不能越界")
        elif not self.original_url:
            raise ValueError("网页/搜索来源必须包含 URL")
        for url in (self.original_url, self.final_url):
            if url:
                parsed = urlsplit(url)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                    raise ValueError("来源 URL 必须为不含凭据的 HTTP/HTTPS 地址")
        return self


class SourceSnapshot(DomainModel):
    account_id: Identifier
    run_id: Identifier
    task_id: Identifier
    source: SourceReference
    body: str = Field(max_length=50000)

    @model_validator(mode="after")
    def matches_evidence(self):
        if hashlib.sha256(self.body.encode("utf-8")).hexdigest() != self.source.content_hash:
            raise ValueError("来源快照哈希不一致")
        start = self.source.evidence_start
        if self.body[start:start + len(self.source.snippet)] != self.source.snippet:
            raise ValueError("引用片段不属于保存的来源快照")
        return self


class SearchOutput(DomainModel):
    """博查发现结果；时间过滤为供应商请求，不能当作热点证明。"""

    query: str
    provider: Identifier = "bocha"
    requested_time_range: TimeRange | None = None
    time_filter_status: Literal["not_requested", "provider_requested"] = "not_requested"
    sources: list[SourceReference] = Field(default_factory=list, max_length=10)
    stored_source_count: StrictInt = Field(ge=0)
    warnings: list[Text] = Field(default_factory=list)
    untrusted_data: Literal[True] = True


class PageOutput(DomainModel):
    """已取得网页的元数据及短证据；完整可见文本快照通过 source_id 回查。"""

    source: SourceReference
    content: str = Field(max_length=4000)
    content_truncated: StrictBool = False
    warnings: list[Text] = Field(default_factory=list)
    untrusted_data: Literal[True] = True


class KnowledgeOutput(DomainModel):
    """当前账号受控 Markdown 的关键词检索结果，不是语义 RAG。"""

    query: str
    sources: list[SourceReference] = Field(default_factory=list, max_length=10)
    stored_source_count: StrictInt = Field(ge=0)
    warnings: list[Text] = Field(default_factory=list)
    untrusted_data: Literal[True] = True


class SearchHit(DomainModel):
    url: str = Field(max_length=2048)
    title: str = Field(max_length=300)
    body: str = Field(max_length=50000)
    published_at_raw: str | None = Field(default=None, max_length=100)
    truncated: StrictBool = False


class SearchDocuments(DomainModel):
    hits: list[SearchHit] = Field(default_factory=list, max_length=10)
    warnings: list[Text] = Field(default_factory=list)


class PageDocument(DomainModel):
    original_url: str
    final_url: str
    title: str = Field(max_length=300)
    body: str = Field(max_length=50000)
    published_at_raw: str | None = Field(default=None, max_length=100)
    truncated: StrictBool = False
    warnings: list[Text] = Field(default_factory=list)


class KnowledgeDocument(DomainModel):
    relative_path: str
    title: str = Field(max_length=300)
    body: str = Field(max_length=50000)
    evidence_start: StrictInt = Field(ge=0)
    score: float = Field(ge=0, le=1)
    truncated: StrictBool = False


class KnowledgeDocuments(DomainModel):
    documents: list[KnowledgeDocument] = Field(default_factory=list, max_length=10)
    warnings: list[Text] = Field(default_factory=list)


class ToolExecution(DomainModel):
    execution_id: Identifier
    run_id: Identifier
    task_id: Identifier
    tool_name: Identifier
    arguments: Payload
    status: Literal["RUNNING", "SUCCEEDED", "FAILED"]
    result: Payload | None = None
    error_code: Identifier | None = None
    attempts: StrictInt = Field(ge=0)
    duration_ms: StrictInt = Field(ge=0)
    created_at: AwareDatetime
    finished_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def consistent_receipt(self):
        if self.status == "RUNNING":
            if self.result is not None or self.finished_at is not None or self.error_code is not None:
                raise ValueError("运行中的工具不能带完成回执")
        else:
            receipt = validate_tool_receipt(self.result)
            expected = "SUCCEEDED" if receipt.ok else "FAILED"
            code = receipt.error_code.value if receipt.error_code else None
            if (self.status, self.tool_name, self.attempts, self.duration_ms, self.error_code) != (
                    expected, receipt.tool, receipt.attempts, receipt.duration_ms, code) or self.finished_at is None:
                raise ValueError("工具记录与执行回执不一致")
        return self


def validate_tool_receipt(payload):
    """复用现有 ToolResult 的成功/失败约束，拒绝不完整或额外字段。"""
    try:
        if type(payload["ok"]) is not bool:
            raise ValueError("ok 必须是布尔值")
        common = dict(tool=payload["tool"], attempts=payload["attempts"], duration_ms=payload["duration_ms"])
        if payload["ok"]:
            result = ToolResult.success(value=payload["value"], **common)
        else:
            error = payload["error"]
            result = ToolResult.failure(error_code=ErrorCode(error["code"]), error_message=error["message"],
                                        retry_exhausted=error["retry_exhausted"], suggestions=error.get("suggestions", ()), **common)
        if result.to_dict() != payload:
            raise ValueError("工具回执字段不一致")
        return result
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("工具回执不符合 ToolResult 契约") from error


def publication_time(raw: str | None) -> datetime | None:
    """只解析带明确时区的 ISO 时间，不猜测日期或本地时区。"""
    if not raw:
        return None
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return value if value.tzinfo is not None and value.utcoffset() is not None else None
    except ValueError:
        return None
