"""提取可见文本、标题和显式发布时间；不执行网页脚本或网页指令。"""

from html.parser import HTMLParser

from media_operations.adapters.http import PublicHTTPClient
from media_operations.research_models import PageDocument, PageQuery
from tooling.errors import ClassifiedToolError
from tooling.result import ErrorCode


class VisiblePageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.title_parts = [], []
        self.ignored_depth = 0
        self.in_title = False
        self.published_at_raw = None

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.ignored_depth += 1
        if tag == "title" and not self.ignored_depth:
            self.in_title = True
        if tag == "meta" and not self.ignored_depth:
            attributes = dict(attrs)
            key = (attributes.get("property") or attributes.get("name") or "").lower()
            if key in {"article:published_time", "datepublished", "date"} and not self.published_at_raw:
                self.published_at_raw = (attributes.get("content") or "")[:100] or None

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.ignored_depth:
            self.ignored_depth -= 1
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        text = " ".join(data.split())
        if text and not self.ignored_depth:
            (self.title_parts if self.in_title else self.parts).append(text)


class WebExtractor:
    def __init__(self, client=None):
        self.client = client or PublicHTTPClient()

    def extract(self, request: PageQuery) -> PageDocument:
        response = self.client.request("GET", request.url)
        if response.content_type not in {"text/html", "text/plain"}:
            raise ClassifiedToolError(ErrorCode.REMOTE_ERROR, "仅支持 HTML 和纯文本网页")
        warnings = []
        try:
            text = response.body.decode(response.charset, errors="replace")
        except LookupError:
            text = response.body.decode("utf-8", errors="replace")
            warnings.append("未知网页编码，已回退 UTF-8")
        title, published = "", None
        if response.content_type == "text/html":
            parser = VisiblePageParser()
            parser.feed(text)
            text, title, published = "\n".join(parser.parts), " ".join(parser.title_parts)[:300], parser.published_at_raw
        truncated = response.truncated or len(text) > 50000
        if truncated:
            warnings.append("网页超过读取上限，保存的是截断快照")
        if not text.strip():
            warnings.append("网页没有可读取的正文")
        if not published:
            warnings.append("未取得发布时间，不能据此判断热点时效")
        return PageDocument(original_url=request.url, final_url=response.url, title=title, body=text[:50000],
                            published_at_raw=published, truncated=truncated, warnings=warnings)
