"""仅检索绑定账号目录下的 Markdown，不暴露工作区文件读取工具。"""

import re
from pathlib import Path

from media_operations.research_models import KnowledgeDocument, KnowledgeDocuments, KnowledgeQuery
from tooling.registry import UnsafeRequestError


class AccountKnowledge:
    def __init__(self, root: Path, account_id: str, *, max_files=200, max_bytes=200000):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", account_id) or max_files < 1 or max_bytes < 1:
            raise UnsafeRequestError("账号知识目录或读取限制无效")
        self.root = Path(root).resolve()
        self.account_directory = self.root / account_id
        if not self.account_directory.resolve().is_relative_to(self.root) or self.account_directory == self.root:
            raise UnsafeRequestError("账号知识目录越界")
        self.max_files, self.max_bytes = max_files, max_bytes

    def search(self, request: KnowledgeQuery) -> KnowledgeDocuments:
        if not self.account_directory.exists():
            return KnowledgeDocuments(warnings=["当前账号尚无 Markdown 知识资料"])
        if self.account_directory.is_symlink() or not self.account_directory.resolve().is_relative_to(self.root):
            raise UnsafeRequestError("知识目录不能通过符号链接越界")
        terms = set(re.findall(r"[a-z0-9_+-]{2,}", request.query.casefold()))
        for word in re.findall(r"[\u4e00-\u9fff]+", request.query):
            terms.add(word)
            terms.update(word[index:index + 2] for index in range(len(word) - 1))
        documents, warnings = [], []
        count = 0
        for entry_count, path in enumerate(self.account_directory.rglob("*")):
            if entry_count >= self.max_files * 10:
                warnings.append("知识目录扫描数量已达上限，结果可能不完整")
                break
            if path.suffix.lower() != ".md" or not path.is_file():
                continue
            count += 1
            if count > self.max_files:
                warnings.append("知识文件数量超过扫描上限，结果可能不完整")
                break
            if path.is_symlink() or not path.resolve().is_relative_to(self.account_directory.resolve()):
                warnings.append("已跳过越界或符号链接文件")
                continue
            # 目录内的符号链接即使指回目录，也不作为可信知识路径。
            if any(parent.is_symlink() for parent in path.parents if parent != self.root and parent.is_relative_to(self.root)):
                warnings.append("已跳过符号链接目录中的知识文件")
                continue
            try:
                with path.open("rb") as stream:
                    raw = stream.read(self.max_bytes + 1)
                text = raw[:self.max_bytes].decode("utf-8")
            except (OSError, UnicodeError):
                warnings.append("已跳过无法读取或非 UTF-8 的 Markdown")
                continue
            truncated = len(raw) > self.max_bytes or len(text) > 50000
            text = text[:50000]
            lowered = text.casefold()
            matches = [term for term in terms if term in lowered]
            if not matches:
                continue
            start = max(0, min(lowered.find(term) for term in matches) - 100)
            title = next((line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")), path.stem)
            documents.append(KnowledgeDocument(
                relative_path=path.relative_to(self.account_directory).as_posix(), title=title[:300],
                body=text, evidence_start=start, score=len(matches) / max(len(terms), 1), truncated=truncated,
            ))
        documents.sort(key=lambda item: (-item.score, item.relative_path))
        if not documents:
            warnings.append("未找到匹配的账号知识资料")
        return KnowledgeDocuments(documents=documents[:request.limit], warnings=list(dict.fromkeys(warnings)))
