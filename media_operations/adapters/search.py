"""沿用博查供应商接入，保留结构化结果和明确的服务错误。"""

import json
import os

from pydantic import ValidationError

from media_operations.adapters.http import PublicHTTPClient
from media_operations.research_models import ResearchQuery, SearchDocuments, SearchHit
from tooling.errors import ClassifiedToolError
from tooling.registry import ToolAuthenticationError
from tooling.result import ErrorCode


class BochaSearch:
    def __init__(self, *, api_key=None, base_url=None, client=None):
        self.api_key = os.getenv("BOCHA_API_KEY") if api_key is None else api_key
        self.base_url = base_url or os.getenv("BOCHA_BASE_URL", "https://api.bochaai.com/v1/web-search")
        self.client = client or PublicHTTPClient(timeout=20, max_bytes=1000000)

    def search(self, request: ResearchQuery) -> SearchDocuments:
        request = ResearchQuery.model_validate(request.model_dump())
        if not self.api_key or "请替换" in self.api_key:
            raise ToolAuthenticationError("请配置 BOCHA_API_KEY 后再检索")
        response = self.client.request("POST", self.base_url, body=json.dumps({
            "query": request.query, "freshness": request.time_range.value if request.time_range else "noLimit",
            "summary": True, "count": request.limit,
        }).encode("utf-8"), headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        if response.truncated:
            raise ClassifiedToolError(ErrorCode.PROTOCOL_ERROR, "搜索响应超过大小上限，未使用部分 JSON")
        try:
            payload = json.loads(response.body)
            code = payload.get("code", 200)
            if code not in {None, 200}:
                error_code = (ErrorCode.AUTHENTICATION_ERROR if code in {401, 403}
                              else ErrorCode.RATE_LIMITED if code == 429
                              else ErrorCode.SERVER_ERROR if isinstance(code, int) and code >= 500 else ErrorCode.REMOTE_ERROR)
                raise ClassifiedToolError(error_code, "搜索服务报告失败（已隐藏供应商原始错误文本）")
            items = payload["data"]["webPages"]["value"]
            if not isinstance(items, list):
                raise ValueError("搜索结果不是列表")
            hits = []
            for item in items[:request.limit]:
                if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                    raise ValueError("搜索结果缺少有效 URL")
                title = item.get("name") or ""
                text = item.get("summary") or item.get("snippet") or ""
                raw_date = item.get("datePublished") or item.get("publishedAt")
                if not isinstance(title, str) or not isinstance(text, str) or (raw_date is not None and not isinstance(raw_date, str)):
                    raise ValueError("搜索字段类型错误")
                hits.append(SearchHit(url=item["url"], title=title[:300], body=text[:50000],
                                      published_at_raw=raw_date[:100] if raw_date else None, truncated=len(text) > 50000))
            warnings = ["搜索摘要仅用于发现线索，未独立验证正文或热点热度"]
            if request.time_range:
                warnings.append("已请求供应商时间过滤；未知发布时间仍为 null，过滤准确性未独立验证")
            if not hits:
                warnings.append("未搜索到相关结果")
            return SearchDocuments(hits=hits, warnings=warnings)
        except (KeyError, ValueError, TypeError, AttributeError, ValidationError) as error:
            raise ClassifiedToolError(ErrorCode.PROTOCOL_ERROR, "搜索服务返回不符合约定的数据") from error
