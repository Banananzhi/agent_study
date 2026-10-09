"""研究工具业务服务：输入校验、来源保存、账号绑定和短证据输出。"""

import hashlib
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import ValidationError

from media_operations.adapters.http import PublicNetworkPolicy
from media_operations.adapters.redaction import Redactor
from media_operations.persistence.errors import ConflictError, NotFoundError
from media_operations.research_models import (
    KnowledgeDocuments, KnowledgeOutput, KnowledgeQuery, PageDocument, PageOutput, PageQuery,
    ResearchQuery, SearchDocuments, SearchOutput, SourceKind, SourceReference,
    SourceSnapshot, publication_time,
)
from media_operations.schemas import RunStatus, TaskStatus, TaskType
from tooling.errors import ClassifiedToolError
from tooling.registry import UnsafeRequestError
from tooling.result import ErrorCode


class ResearchService:
    def __init__(self, repository, account_id, run_id, task_id, *, search, web, knowledge, redactor=None, policy=None):
        self.repository = repository
        self.account_id, self.run_id, self.task_id = account_id, run_id, task_id
        self.search, self.web, self.knowledge = search, web, knowledge
        self.redactor = redactor or Redactor()
        self.policy = policy or PublicNetworkPolicy()
        self.check_active()

    def check_active(self):
        try:
            run = self.repository.get_run(self.account_id, self.run_id)
            task = self.repository.get_task(self.account_id, self.run_id, self.task_id)
        except NotFoundError as error:
            raise ClassifiedToolError(ErrorCode.POLICY_DENIED, "研究上下文不可访问") from error
        if run.status != RunStatus.RUNNING or task.status != TaskStatus.RUNNING or task.task_type != TaskType.RESEARCH:
            raise ClassifiedToolError(ErrorCode.POLICY_DENIED, "研究任务已停止或当前任务不是 research")

    @staticmethod
    def _input(model, arguments):
        try:
            return model.model_validate(arguments)
        except ValidationError as error:
            first = error.errors(include_input=False, include_url=False)[0]
            raise ValueError(f"研究工具参数不符合约定：{'.'.join(map(str, first['loc']))}") from error

    def _snapshot(self, *, kind, provider, body, title, original_url=None, final_url=None,
                  relative_path=None, raw_date=None, start=0, truncated=False):
        # 来源中疑似凭据也先脱敏再保存；哈希对应实际保存的文本。
        evidence_start = len(self.redactor.text(body[:start]))
        redacted_body = self.redactor.text(body)
        truncated = truncated or len(redacted_body) > 50000
        body = redacted_body[:50000]
        raw_date = self.redactor.text(raw_date) if raw_date else None
        start = min(evidence_start, len(body))
        source = SourceReference(
            source_id=f"source_{uuid4().hex}", kind=kind, provider=provider,
            original_url=self.redactor.text(original_url) if original_url else None,
            final_url=self.redactor.text(final_url) if final_url else None,
            relative_path=self.redactor.text(relative_path) if relative_path else None,
            title=self.redactor.text(title)[:300], published_at=publication_time(raw_date), published_at_raw=raw_date[:100] if raw_date else None,
            retrieved_at=datetime.now(UTC), snippet=body[start:start + 600], evidence_start=start,
            content_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(), truncated=truncated,
            verification_status="discovered" if kind == SourceKind.SEARCH else "retrieved",
        )
        return SourceSnapshot(account_id=self.account_id, run_id=self.run_id, task_id=self.task_id, source=source, body=body)

    def _save(self, snapshots):
        try:
            return self.repository.save_sources(self.account_id, self.run_id, self.task_id, snapshots)
        except (ConflictError, NotFoundError) as error:
            raise ClassifiedToolError(ErrorCode.POLICY_DENIED, "研究上下文已变化，来源结果未提交") from error

    @staticmethod
    def _compact(output):
        # 保留完整来源条目，避免框架把超大列表截断成不可解析 JSON；所有来源已落库。
        sources = list(output.sources)
        warnings = list(output.warnings)
        while sources and len(output.model_dump_json()) > 6000:
            sources.pop()
            output = type(output).model_validate({**output.model_dump(), "sources": sources})
        if len(sources) < output.stored_source_count:
            warnings.append("部分来源因上下文大小省略；完整快照已保存，可从当前 Run 查询")
        return type(output).model_validate({**output.model_dump(), "warnings": warnings})

    def search_web(self, **arguments):
        self.check_active()
        request = self._input(ResearchQuery, arguments)
        documents = SearchDocuments.model_validate(self.search.search(request).model_dump())
        snapshots, warnings = [], [self.redactor.text(item) for item in documents.warnings]
        for hit in documents.hits:
            try:
                self.policy.validate_url(hit.url)
            except UnsafeRequestError:
                warnings.append("已跳过不符合网络策略的搜索来源")
                continue
            snapshots.append(self._snapshot(kind=SourceKind.SEARCH, provider="bocha", body=hit.body,
                                            title=hit.title, original_url=hit.url, raw_date=hit.published_at_raw,
                                            truncated=hit.truncated))
        stored = self._save(snapshots)
        if not stored:
            warnings.append("未取得可用搜索来源，不得据此编造热点")
        return self._compact(SearchOutput(
            query=self.redactor.text(request.query), requested_time_range=request.time_range,
            time_filter_status="provider_requested" if request.time_range else "not_requested",
            sources=[item.source for item in stored], stored_source_count=len(stored), warnings=list(dict.fromkeys(warnings)),
        )).model_dump(mode="json")

    def extract_web_page(self, **arguments):
        self.check_active()
        request = self._input(PageQuery, arguments)
        self.policy.validate_url(request.url)
        document = PageDocument.model_validate(self.web.extract(request).model_dump())
        self.policy.validate_url(document.final_url)
        snapshot = self._snapshot(kind=SourceKind.WEB, provider="public_http", body=document.body,
                                  title=document.title, original_url=request.url, final_url=document.final_url,
                                  raw_date=document.published_at_raw, truncated=document.truncated)
        stored = self._save([snapshot])[0]
        warnings = list(document.warnings)
        if document.published_at_raw and stored.source.published_at is None:
            warnings.append("发布时间格式或时区不明确，保留原值但标准时间为 null")
        content = stored.body[:4000]
        output = PageOutput(source=stored.source, content=content, content_truncated=len(content) < len(stored.body), warnings=warnings)
        excess = len(output.model_dump_json()) - 6000
        if excess > 0:
            content = content[:max(0, len(content) - excess - 100)]
            output = PageOutput(source=stored.source, content=content, content_truncated=True, warnings=warnings)
        if output.content_truncated:
            output = PageOutput(**{**output.model_dump(), "warnings": warnings + ["正文工作上下文已裁剪，完整保存文本通过 source_id 回查"]})
        return PageOutput.model_validate(self.redactor.value(output.model_dump(mode="json"))).model_dump(mode="json")

    def search_account_knowledge(self, **arguments):
        self.check_active()
        request = self._input(KnowledgeQuery, arguments)
        if request.account_id is not None and request.account_id != self.account_id:
            raise UnsafeRequestError("知识查询不能切换绑定账号")
        documents = KnowledgeDocuments.model_validate(self.knowledge.search(request).model_dump())
        snapshots = [self._snapshot(kind=SourceKind.KNOWLEDGE, provider="account_markdown", body=item.body,
                                    title=item.title, relative_path=item.relative_path, start=item.evidence_start,
                                    truncated=item.truncated) for item in documents.documents]
        stored = self._save(snapshots)
        return self._compact(KnowledgeOutput(
            query=self.redactor.text(request.query), sources=[item.source for item in stored],
            stored_source_count=len(stored), warnings=[self.redactor.text(item) for item in documents.warnings],
        )).model_dump(mode="json")
