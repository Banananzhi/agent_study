import json
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

import httpx
from langchain_core.messages import AIMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from qdrant_client import QdrantClient

from agent import Agent
from agent.memory.extractor import StructuredMemoryExtractor
from agent.memory.models import MemoryDecision
from agent.memory.reconciliation import MemoryReconciler
from agent.memory.resolver import MemoryConflictResolver
from agent.memory import (
    IndexJobStatus,
    IndexStatus,
    LongTermMemoryService,
    MemoryRepository,
    MemoryRetryPolicy,
    MemorySearchHit,
    MemoryCandidate,
    MemoryExtractionBatch,
    MemoryType,
    MemoryWrite,
    OpenAICompatibleEmbeddingProvider,
    QdrantMemoryIndex,
)


class StructuredMemoryExtractorTest(unittest.TestCase):
    # 验证真实请求关闭提取推理、保留强制工具输出，且不影响主模型配置
    def test_extraction_disables_thinking_without_mutating_main_model(self):
        # captured：模拟服务收到的实际 HTTP 请求体
        captured = []

        # 返回合法的结构化工具响应，验证请求编码与 Pydantic 解析的完整链路
        # request：LangChain 经 OpenAI 客户端发出的请求
        def handler(request):
            captured.append(json.loads(request.content))
            # result：根据绑定的 Schema 返回对应结构，覆盖提取器与判断器
            result = (MemoryDecision(action="DEFER", reason="测试判断结果")
                      if captured[-1]["tools"][0]["function"]["name"] == "MemoryDecision"
                      else MemoryExtractionBatch())
            return httpx.Response(200, json={
                "id": "test-extraction",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-flash",
                "choices": [{
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call-memory",
                            "type": "function",
                            "function": {
                                "name": type(result).__name__,
                                "arguments": result.model_dump_json(),
                            },
                        }],
                    },
                }],
            })

        # client：拦截 HTTP 请求，测试无需真实密钥或网络
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            # model：模拟主 Agent 开启推理并带有其他扩展参数
            model = ChatDeepSeek(
                model="deepseek-flash",
                api_key="test-key",
                http_client=client,
                max_retries=0,
                extra_body={"thinking": {"type": "enabled"}, "test_option": True},
            )
            # extractor：独立配置的记忆提取器
            extractor = StructuredMemoryExtractor(model)
            self.assertIsInstance(extractor.extract("你好", "你好", []), MemoryExtractionBatch)
            self.assertEqual({"type": "disabled"}, captured[0]["thinking"])
            self.assertTrue(captured[0]["test_option"])
            self.assertEqual(
                {"type": "function", "function": {"name": "MemoryExtractionBatch"}},
                captured[0]["tool_choice"],
            )
            self.assertEqual({"type": "enabled"}, model.extra_body["thinking"])
            # 再次通过主模型发送请求，确认 HTTP 层仍使用原有推理配置
            model.bind_tools([MemoryExtractionBatch], tool_choice="auto").invoke("你好")
            self.assertEqual({"type": "enabled"}, captured[1]["thinking"])
            self.assertEqual("auto", captured[1]["tool_choice"])
            # resolver：同样禁用 thinking，且请求携带候选和来源原文
            resolver = MemoryConflictResolver(model)
            # candidate：仅用于验证判断请求编码与结构化解析
            candidate = MemoryCandidate(content="偏好短文案", memory_type="preference",
                                        importance=0.9, confidence=1.0)
            self.assertEqual("DEFER", resolver.resolve(candidate, [], "测试原文").action)
            self.assertEqual({"type": "disabled"}, captured[2]["thinking"])
            self.assertEqual("MemoryDecision", captured[2]["tool_choice"]["function"]["name"])
            self.assertIn("测试原文", captured[2]["messages"][-1]["content"])
            self.assertEqual({"type": "enabled"}, model.extra_body["thinking"])


class FakeMemoryExtractor:
    # 初始化返回固定候选批次的测试记忆提取器
    # batch：每次提取返回的结构化候选批次
    # error：需要模拟的提取异常
    def __init__(self, batch=None, error=None):
        self.batch = batch or MemoryExtractionBatch()
        self.error = error
        self.calls = []

    # 记录当前完整轮次并返回固定候选记忆
    # user_message：本轮用户输入
    # final_answer：本轮最终答案
    # tool_observations：本轮经过长度治理的工具观察
    def extract(self, user_message, final_answer, tool_observations):
        self.calls.append((user_message, final_answer, tool_observations))
        if self.error is not None:
            raise self.error
        return self.batch


class MemoryScriptedAgent(Agent):
    # 初始化使用固定最终回答的长期记忆集成测试 Agent
    # responses：模型应按顺序返回的 AIMessage 列表
    # attrs：传递给 Agent 的其他配置
    def __init__(self, responses, **attrs):
        super().__init__(**attrs)
        self.responses = iter(responses)
        self.seen_messages = []

    # 记录注入长期记忆后的模型输入并返回固定响应
    # messages：当前模型实际收到的上下文消息
    def think(self, messages):
        self.seen_messages.append(list(messages))
        return next(self.responses)


class FakeEmbedder:
    # 初始化无需下载模型的确定性测试向量组件
    def __init__(self):
        self.model_name = "test-embedding"
        self.documents = []
        self.queries = []

    # 根据文本长度生成可预测的测试文档向量
    # texts：需要记录并生成向量的文本列表
    def embed_documents(self, texts):
        self.documents.extend(texts)
        return [[float(len(text)), 1.0, 0.5] for text in texts]

    # 根据查询长度生成可预测的测试查询向量
    # text：需要记录并生成向量的查询文本
    def embed_query(self, text):
        self.queries.append(text)
        return [float(len(text)), 1.0, 0.5]


class OnlineEmbeddingProviderTest(unittest.TestCase):
    # 验证在线Embedding请求使用指定模型、鉴权并恢复输入顺序
    def test_openai_compatible_embedding_request(self):
        # captured：MockTransport 记录的请求体和鉴权信息
        captured = {}

        # 返回故意打乱下标顺序的标准 OpenAI Embedding 响应
        # request：Provider 发出的 HTTP 请求
        def handler(request):
            captured["authorization"] = request.headers["Authorization"]
            captured["body"] = request.content.decode("utf-8")
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ]
                },
            )

        # client：不会访问真实百炼服务的测试HTTP客户端
        client = httpx.Client(transport=httpx.MockTransport(handler))
        # provider：使用用户指定模型的在线向量组件
        provider = OpenAICompatibleEmbeddingProvider(
            model_name="qwen3.7-text-embedding-flash",
            base_url="https://embedding.example/v1",
            api_key="test-key",
            client=client,
        )
        try:
            vectors = provider.embed_documents(["第一条", "第二条"])
        finally:
            client.close()

        self.assertEqual([[1.0, 0.0], [0.0, 1.0]], vectors)
        self.assertEqual("Bearer test-key", captured["authorization"])
        self.assertIn('"model":"qwen3.7-text-embedding-flash"', captured["body"])

    # 验证缺少在线Embedding密钥时在发起网络请求前明确报错
    def test_embedding_api_key_is_required(self):
        # provider：没有配置真实密钥的在线向量组件
        provider = OpenAICompatibleEmbeddingProvider(api_key=None)
        try:
            with self.assertRaisesRegex(RuntimeError, "AGENT_EMBEDDING_API_KEY"):
                provider.embed_query("测试")
        finally:
            provider.close()


class FakeMemoryIndex:
    # 初始化可记录调用并模拟异常的测试向量索引
    # error：执行 upsert 时需要抛出的异常
    def __init__(self, error=None):
        self.error = error
        self.upserts = []
        self.deletes = []
        self.search_hits = []
        self.search_args = None

    # 记录幂等向量写入
    # memory：准备建立索引的长期记忆
    # vector：测试 Embedding 生成的向量
    def upsert(self, memory, vector):
        if self.error is not None:
            raise self.error
        self.upserts.append((memory, vector))

    # 记录幂等向量删除
    # memory_id：需要删除的长期记忆标识
    def delete(self, memory_id):
        self.deletes.append(memory_id)

    # 返回预先配置的语义检索结果
    # vector：查询向量
    # tenant_id：可信租户标识
    # user_id：可信用户标识
    # attrs：其他检索过滤参数
    def search(self, vector, tenant_id, user_id, **attrs):
        self.search_args = (vector, tenant_id, user_id, attrs)
        return self.search_hits


class MemoryServiceTest(unittest.TestCase):
    # 为每个测试创建互不影响的临时 SQLite 数据库
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # database_path：当前测试使用的 SQLite 文件
        database_path = Path(self.temp_dir.name) / "memories.sqlite3"
        self.repository = MemoryRepository(database_path)
        self.embedder = FakeEmbedder()
        self.index = FakeMemoryIndex()
        self.service = LongTermMemoryService(
            self.repository,
            self.embedder,
            self.index,
            retry_policy=MemoryRetryPolicy(max_attempts=3),
        )

    # 清理测试创建的临时数据库目录
    def tearDown(self):
        self.temp_dir.cleanup()

    # 创建当前测试复用的用户偏好写入请求
    # content：记忆完整文本
    # memory_key：支持更新去重的稳定键
    # user_id：记忆所属用户
    def memory_write(
        self,
        content="用户喜欢简洁的技术解释",
        memory_key="explanation_style",
        user_id="user-1",
    ):
        return MemoryWrite(
            tenant_id="tenant-1",
            user_id=user_id,
            project_id="agent-study",
            memory_type=MemoryType.PREFERENCE,
            memory_key=memory_key,
            content=content,
            importance=0.8,
            confidence=0.9,
            source_thread_id="thread-1",
            source_message_ids=["message-1"],
        )

    # 验证记忆和待索引任务由一次 remember 同时创建
    def test_remember_creates_memory_and_outbox_job(self):
        memory = self.service.remember(self.memory_write())

        self.assertEqual(IndexStatus.PENDING, memory.index_status)
        # jobs：与新记忆一起提交的 Outbox 任务
        jobs = self.repository.list_jobs(memory.id)
        self.assertEqual(1, len(jobs))
        self.assertEqual(IndexJobStatus.PENDING, jobs[0].status)
        self.assertEqual(memory.version, jobs[0].memory_version)

    # 生成带原文证据的记忆候选，便于复用各类冲突测试
    # content：候选正文；attrs：要覆盖的来源、意图或范围等字段
    def candidate(self, content="文案最多80字", **attrs):
        return MemoryCandidate(**{
            "content": content, "memory_type": "preference", "memory_key": "copy_limit",
            "importance": 0.9, "confidence": 1.0, "source_kind": "user",
            "evidence": "以后文案最多80字", "change_intent": "update", **attrs,
        })

    # 执行冲突管线，替身判断器避免产生真实网络请求
    # candidate：候选记忆；resolver：固定或动态返回决策的替身
    # attrs：可信范围的可选覆盖字段
    def reconcile(self, candidate, resolver, **attrs):
        # value：将候选正文放入默认测试范围
        value = self.memory_write(candidate.content, candidate.memory_key).model_copy(update=attrs)
        return MemoryReconciler(self.service, resolver).reconcile(
            value, candidate, "以后文案最多80字", [],
        )

    # 验证相同正文跨键重复不会增加版本、历史或 Outbox
    def test_exact_duplicate_skips_model_and_index_job(self):
        # original：尚未同步向量的旧事实
        original = self.service.remember(self.memory_write("文案最多80字", "old_key"))
        # resolver：重复规则应该完全绕过模型
        resolver = Mock()
        self.assertEqual("NOOP", self.reconcile(self.candidate(), resolver).action)
        resolver.resolve.assert_not_called()
        self.assertEqual(1, self.repository.get(original.id).version)
        self.assertEqual(1, len(self.repository.list_jobs(original.id)))
        self.service.remember(self.memory_write("文案最多80字", "old_key"))
        self.assertEqual(1, len(self.repository.list_jobs(original.id)))

    # 验证语义重复不更新正文，即使模型生成了不同稳定键
    def test_semantic_duplicate_preserves_original(self):
        # original：旧偏好及固定 NOOP 判断器
        original = self.service.remember(self.memory_write("文案上限为八十字", "old_key"))
        # resolver：模拟语义相同的判断
        resolver = Mock(resolve=Mock(return_value=MemoryDecision(
            action="NOOP", target_id=original.id, reason="相同字数限制",
        )))
        self.assertEqual("NOOP", self.reconcile(self.candidate(), resolver).action)
        self.assertEqual(1, self.repository.get(original.id).version)

    # 验证明示修改保持 ID 与旧稳定键，归档旧正文并通过 Worker 索引新版本
    def test_explicit_update_archives_history_and_indexes_new_version(self):
        # original：已索引的旧字数限制
        original = self.service.remember(self.memory_write("文案最多45字", "old_key"))
        self.service.process_pending()
        # resolver：候选新键仍然指向同一个旧事实
        resolver = Mock(resolve=Mock(return_value=MemoryDecision(
            action="UPDATE", target_id=original.id, reason="用户明确修改长期限制",
        )))
        self.assertEqual("UPDATE", self.reconcile(self.candidate(), resolver).action)
        # updated：更新保持稳定标识和旧键，版本递增
        updated = self.repository.get(original.id)
        self.assertEqual(("文案最多80字", 2, "old_key"),
                         (updated.content, updated.version, updated.memory_key))
        with self.repository._connect() as connection:
            # history：与新正文、决策、任务原子保存的旧快照
            history = connection.execute("SELECT snapshot_json FROM memory_versions").fetchone()
            self.assertEqual("文案最多45字", json.loads(history[0])["content"])
            self.assertEqual("UPDATE", connection.execute("SELECT action FROM memory_decisions").fetchone()[0])
        self.assertEqual(1, self.service.process_pending().completed)
        self.assertEqual("文案最多80字", self.index.upserts[-1][0].content)

    # 验证临时要求、伪造证据和助手复述不会改变用户长期偏好
    def test_temporary_or_unverified_sources_are_deferred(self):
        # original：需要保护的长期偏好
        original = self.service.remember(self.memory_write("文案最多45字", "copy_limit"))
        # attrs：各类不允许更新的来源
        for attrs in ({"scope_kind": "temporary"}, {"evidence": "不存在的用户原文"},
                      {"source_kind": "assistant"}):
            with self.subTest(attrs=attrs):
                # resolver：预检查失败不应调用判断模型
                resolver = Mock()
                self.assertEqual("DEFER", self.reconcile(self.candidate(**attrs), resolver).action)
                resolver.resolve.assert_not_called()
        self.assertEqual(1, self.repository.get(original.id).version)

    # 验证模型建议不能越过明确修改意图与工具证据限制
    def test_update_requires_explicit_user_intent(self):
        # original：用于模型更新建议的旧记忆
        original = self.service.remember(self.memory_write("文案最多45字", "copy_limit"))
        # resolver：模拟过于激进的更新建议
        resolver = Mock(resolve=Mock(return_value=MemoryDecision(
            action="UPDATE", target_id=original.id, reason="候选值不同",
        )))
        self.assertEqual("DEFER", self.reconcile(self.candidate(change_intent="unspecified"), resolver).action)
        # candidate：工具事实只能新增，不能代替用户授权覆盖偏好
        candidate = self.candidate(source_kind="tool")
        self.assertEqual("DEFER", MemoryReconciler(self.service, resolver).reconcile(
            self.memory_write(candidate.content, candidate.memory_key), candidate,
            "以后文案最多80字", [candidate.evidence],
        ).action)
        self.assertEqual(1, self.repository.get(original.id).version)

    # 验证返回陌生 ID 或其他用户、租户、项目的 ID 时只能暂缓
    def test_foreign_targets_cannot_be_updated(self):
        self.service.remember(self.memory_write("文案最多45字", "local"))
        # attrs：三个独立隔离边界
        for attrs in ({"tenant_id": "other"}, {"user_id": "other"}, {"project_id": "other"}):
            with self.subTest(attrs=attrs):
                # foreign：位于不同身份或项目范围的旧记忆
                foreign = self.service.remember(self.memory_write("他人的记忆", "foreign").model_copy(update=attrs))
                # resolver：模拟返回越界目标的错误模型
                resolver = Mock(resolve=Mock(return_value=MemoryDecision(
                    action="UPDATE", target_id=foreign.id, reason="错误目标",
                )))
                self.assertEqual("DEFER", self.reconcile(self.candidate(), resolver).action)
                self.assertEqual(1, self.repository.get(foreign.id).version)

    # 验证 ADD 不能通过相同稳定键绕过 UPDATE 的明确授权要求
    def test_add_cannot_silently_overwrite_key(self):
        # original：与候选同键但内容不同
        original = self.service.remember(self.memory_write("文案最多45字", "copy_limit"))
        # resolver：模拟错误的新增决策
        resolver = Mock(resolve=Mock(return_value=MemoryDecision(action="ADD", reason="新增")))
        self.assertEqual("DEFER", self.reconcile(self.candidate(), resolver).action)
        self.assertEqual(1, self.repository.get(original.id).version)

    # 验证模型判断期间的并发更新不会被过期决策覆盖
    def test_concurrent_update_defers_stale_decision(self):
        # original：最初读取到的旧记忆
        original = self.service.remember(self.memory_write("文案最多45字", "copy_limit"))

        # 在模型返回前模拟另一个会话写入新版本
        # args：判断器接收的候选、旧记忆和原文
        def decide(*args):
            self.service.remember(self.memory_write("文案最多100字", "copy_limit"))
            return MemoryDecision(action="UPDATE", target_id=original.id, reason="修改到80字")

        # resolver：通过回调复现模型请求期间的并发交错
        resolver = Mock(resolve=Mock(side_effect=decide))
        self.assertEqual("DEFER", self.reconcile(self.candidate(), resolver).action)
        self.assertEqual("文案最多100字", self.repository.get(original.id).content)

    # 验证判断器失败会被记录为暂缓，不会修改记忆或创建同步任务
    def test_resolver_failure_is_audited_as_defer(self):
        self.service.remember(self.memory_write("文案最多45字", "copy_limit"))
        # resolver：模拟模型超时
        resolver = Mock(resolve=Mock(side_effect=TimeoutError("请求超时")))
        self.assertEqual("DEFER", self.reconcile(self.candidate(), resolver).action)
        with self.repository._connect() as connection:
            self.assertEqual("DEFER", connection.execute("SELECT action FROM memory_decisions").fetchone()[0])
        self.assertEqual(1, len(self.repository.list_jobs()))

    # 验证 Outbox 创建失败时正文、历史和决策全部回滚
    def test_update_transaction_rolls_back_on_outbox_failure(self):
        # original：应在失败后完整保留的旧记忆
        original = self.service.remember(self.memory_write("文案最多45字", "copy_limit"))
        # resolver：合法更新决策
        resolver = Mock(resolve=Mock(return_value=MemoryDecision(
            action="UPDATE", target_id=original.id, reason="明确更新",
        )))
        with patch.object(self.repository, "_insert_job", side_effect=RuntimeError("写入失败")):
            with self.assertRaises(RuntimeError):
                self.reconcile(self.candidate(), resolver)
        self.assertEqual("文案最多45字", self.repository.get(original.id).content)
        with self.repository._connect() as connection:
            self.assertEqual(0, connection.execute("SELECT count(*) FROM memory_versions").fetchone()[0])
            self.assertEqual(0, connection.execute("SELECT count(*) FROM memory_decisions").fetchone()[0])

    # 验证大范围语义检索失败时不能被当成没有旧记忆而新增
    def test_semantic_search_failure_defers_candidate(self):
        # index：创建足够多的旧记忆进入语义检索路径
        for index in range(17):
            self.service.remember(self.memory_write(f"旧事实{index}", f"key_{index}"))
        self.service.process_pending()
        self.service.process_pending()
        with patch.object(self.service, "recall", side_effect=ConnectionError("Qdrant不可用")):
            self.assertEqual("DEFER", self.reconcile(self.candidate(), Mock()).action)

    # 验证大范围检索合并未索引旧记忆，且严格限制项目
    def test_pending_memory_included_in_semantic_context(self):
        # index：已索引的旧记忆，触发语义搜索路径
        for index in range(17):
            self.service.remember(self.memory_write(f"旧事实{index}", f"key_{index}"))
        self.service.process_pending()
        self.service.process_pending()
        # pending：刚写入但还没有向量，必须交给判断模型
        pending = self.service.remember(self.memory_write("文案上限45字", "old_limit"))
        # resolver：新旧键不同也可以基于待索引正文识别冲突
        resolver = Mock(resolve=Mock(return_value=MemoryDecision(
            action="UPDATE", target_id=pending.id, reason="用户明确更新",
        )))
        self.assertEqual("UPDATE", self.reconcile(self.candidate(), resolver).action)
        self.assertIn(pending.id, [item.id for item in resolver.resolve.call_args.args[1]])
        self.assertTrue(self.index.search_args[3]["exact_project"])

    # 验证已领取的旧版本任务在新正文提交后不会向索引写入过期内容
    def test_claimed_old_job_skips_after_memory_update(self):
        # original：待索引旧记忆；job：模拟旧 Worker 已领取的任务
        original = self.service.remember(self.memory_write("文案最多45字", "copy_limit"))
        job = self.repository.claim_jobs()[0]
        # resolver：当前轮次生成有效更新
        resolver = Mock(resolve=Mock(return_value=MemoryDecision(
            action="UPDATE", target_id=original.id, reason="明确更新",
        )))
        self.assertEqual("UPDATE", self.reconcile(self.candidate(), resolver).action)
        self.assertEqual(IndexJobStatus.SUPERSEDED, self.service._process_job(job))
        self.assertEqual([], self.index.upserts)
        self.assertEqual(1, self.service.process_pending().completed)
        self.assertEqual("文案最多80字", self.index.upserts[-1][0].content)

    # 验证 Worker 成功生成向量、写入索引并更新 SQLite 状态
    def test_process_pending_indexes_memory(self):
        memory = self.service.remember(self.memory_write())

        report = self.service.process_pending()

        self.assertEqual(1, report.completed)
        self.assertEqual(memory.id, self.index.upserts[0][0].id)
        self.assertEqual(IndexStatus.INDEXED, self.repository.get(memory.id).index_status)
        self.assertEqual(
            IndexJobStatus.COMPLETED,
            self.repository.list_jobs(memory.id)[0].status,
        )

    # 验证相同稳定键更新原记忆并使旧的待执行任务失效
    def test_keyed_memory_updates_version_and_supersedes_old_job(self):
        original = self.service.remember(self.memory_write())
        updated = self.service.remember(
            self.memory_write(content="用户喜欢简洁、重点明确的技术解释")
        )

        self.assertEqual(original.id, updated.id)
        self.assertEqual(2, updated.version)
        # jobs：同一条记忆的历史索引任务
        jobs = self.repository.list_jobs(original.id)
        self.assertEqual(IndexJobStatus.SUPERSEDED, jobs[0].status)
        self.assertEqual(IndexJobStatus.PENDING, jobs[1].status)

        report = self.service.process_pending()
        self.assertEqual(1, report.completed)
        self.assertEqual(updated.content, self.index.upserts[0][0].content)

    # 验证临时基础设施异常保留任务并安排后续重试
    def test_retryable_index_error_keeps_pending_job(self):
        self.index.error = OSError("Qdrant暂时不可用")
        memory = self.service.remember(self.memory_write())

        report = self.service.process_pending()

        self.assertEqual(1, report.retried)
        # job：执行失败后重新进入 pending 的索引任务
        job = self.repository.list_jobs(memory.id)[0]
        self.assertEqual(IndexJobStatus.PENDING, job.status)
        self.assertEqual(1, job.attempts)
        self.assertIsNotNone(job.next_retry_at)

    # 验证不可恢复的向量格式错误进入死信并保留错误信息
    def test_non_retryable_index_error_becomes_dead(self):
        self.index.error = ValueError("向量维度错误")
        memory = self.service.remember(self.memory_write())

        report = self.service.process_pending()

        self.assertEqual(1, report.dead)
        # job：不可恢复错误对应的死信任务
        job = self.repository.list_jobs(memory.id)[0]
        self.assertEqual(IndexJobStatus.DEAD, job.status)
        self.assertIn("向量维度错误", job.last_error)
        self.assertEqual(IndexStatus.FAILED, self.repository.get(memory.id).index_status)

    # 验证软删除通过 Outbox 同步删除 Qdrant Point
    def test_forget_creates_and_processes_delete_job(self):
        memory = self.service.remember(self.memory_write())
        self.service.process_pending()

        deleted = self.service.forget(memory.id, "tenant-1", "user-1")
        report = self.service.process_pending()

        self.assertTrue(deleted)
        self.assertEqual(1, report.completed)
        self.assertEqual([memory.id], self.index.deletes)

    # 验证 Qdrant 命中只能读取可信租户和用户范围内的 SQLite 正文
    def test_recall_enforces_user_scope_when_loading_text(self):
        allowed = self.service.remember(self.memory_write(user_id="user-1"))
        blocked = self.service.remember(
            self.memory_write(memory_key="other", user_id="user-2")
        )
        self.index.search_hits = [
            MemorySearchHit(memory_id=blocked.id, score=0.99),
            MemorySearchHit(memory_id=allowed.id, score=0.80),
        ]
        # 测试中使用替身 Qdrant，需要先模拟两个 SQLite 版本已经完成索引
        self.service.process_pending()

        recalled = self.service.recall("我喜欢怎样的解释", "tenant-1", "user-1", project_id="agent-study")

        self.assertEqual([(allowed.id, 0.80)], [(item.id, score) for item, score in recalled])
        self.assertEqual("tenant-1", self.index.search_args[1])
        self.assertEqual("user-1", self.index.search_args[2])

    # 验证尚未创建Qdrant Collection时删除未索引记忆也能幂等完成
    def test_delete_before_first_index_is_safe(self):
        memory = self.service.remember(self.memory_write())
        deleted = self.service.forget(memory.id, "tenant-1", "user-1")

        report = self.service.process_pending()

        self.assertTrue(deleted)
        self.assertEqual(1, report.completed)
        self.assertEqual([memory.id], self.index.deletes)

    # 验证同一个用户轮次只能领取一次有效记忆提取执行权
    def test_extraction_turn_is_idempotent(self):
        first_claim = self.service.begin_extraction(
            "thread-1:message-1",
            "thread-1",
            "message-1",
        )
        duplicate_claim = self.service.begin_extraction(
            "thread-1:message-1",
            "thread-1",
            "message-1",
        )
        self.service.complete_extraction("thread-1:message-1", 0)
        completed_claim = self.service.begin_extraction(
            "thread-1:message-1",
            "thread-1",
            "message-1",
        )

        self.assertTrue(first_claim)
        self.assertFalse(duplicate_claim)
        self.assertFalse(completed_claim)

    # 验证 Agent 每轮开始召回记忆并在最终答案后写入提取候选
    def test_agent_recall_and_extraction_are_connected_to_graph(self):
        existing = self.service.remember(
            MemoryWrite(
                tenant_id="tenant-1",
                user_id="user-1",
                project_id="agent-study",
                memory_type=MemoryType.PREFERENCE,
                memory_key="copy_style",
                content="用户偏好简洁、重点明确的文案。",
                importance=0.9,
                confidence=1.0,
            )
        )
        self.service.process_pending()
        self.index.search_hits = [MemorySearchHit(memory_id=existing.id, score=0.91)]
        # extractor：本轮最终答案后返回一条新的项目决策
        extractor = FakeMemoryExtractor(
            MemoryExtractionBatch(
                candidates=[
                    MemoryCandidate(
                        memory_type=MemoryType.DECISION,
                        memory_key="memory_storage",
                        content="项目长期记忆使用 SQLite 保存文本并使用 Qdrant 保存向量。",
                        importance=0.9,
                        confidence=1.0,
                        evidence="长期记忆使用 SQLite 保存文本并使用 Qdrant 保存向量",
                        source_kind="user",
                        change_intent="new",
                    )
                ]
            )
        )
        # agent：使用固定回答避免测试访问真实模型
        agent = MemoryScriptedAgent(
            [AIMessage(content="已经确认长期记忆方案。", id="assistant-final")],
            memory_service=self.service,
            memory_extractor=extractor,
            memory_resolver=Mock(resolve=Mock(return_value=MemoryDecision(action="ADD", reason="新增存储决策"))),
            tenant_id="tenant-1",
            user_id="user-1",
            project_id="agent-study",
        )

        answer = agent.run("长期记忆使用 SQLite 保存文本并使用 Qdrant 保存向量")

        self.assertEqual("已经确认长期记忆方案。", answer)
        # memory_messages：只注入本次模型输入的临时长期记忆系统消息
        memory_messages = [
            message
            for message in agent.seen_messages[0]
            if isinstance(message, SystemMessage)
            and message.additional_kwargs.get("long_term_memory")
        ]
        self.assertEqual(1, len(memory_messages))
        self.assertIn("用户偏好简洁", memory_messages[0].content)
        self.assertEqual(1, len(extractor.calls))
        # stored：最终答案生成后写入 SQLite 的新长期记忆
        stored = self.repository.get_by_key(
            "tenant-1",
            "user-1",
            "agent-study",
            MemoryType.DECISION,
            "memory_storage",
        )
        self.assertIsNotNone(stored)
        self.assertEqual(IndexStatus.PENDING, stored.index_status)

    # 验证附加的记忆提取失败不会覆盖或阻断模型最终答案
    def test_extraction_failure_does_not_fail_agent_answer(self):
        # extractor：模拟结构化提取模型临时失败
        extractor = FakeMemoryExtractor(error=RuntimeError("提取模型暂时不可用"))
        # agent：不需要真实模型和Qdrant即可验证降级路径
        agent = MemoryScriptedAgent(
            [AIMessage(content="主任务答案", id="assistant-final")],
            memory_service=self.service,
            memory_extractor=extractor,
            tenant_id="tenant-1",
            user_id="user-1",
        )

        answer = agent.run("完成主任务")

        self.assertEqual("主任务答案", answer)
        self.assertEqual(1, len(extractor.calls))


class QdrantMemoryIndexTest(unittest.TestCase):
    # 验证真实 Qdrant Client 的本地模式能够写入、检索并隔离用户
    def test_qdrant_client_round_trip_and_user_filter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # repository：用于构造完整长期记忆模型的临时 Repository
            repository = MemoryRepository(Path(temp_dir) / "memory.sqlite3")
            # memory：准备写入本地 Qdrant 测试实例的记忆
            memory = repository.upsert(
                MemoryWrite(
                    tenant_id="tenant-1",
                    user_id="user-1",
                    memory_type=MemoryType.EPISODIC,
                    content="用户过去对折叠屏手机营销内容更感兴趣",
                ),
                "test-embedding",
            )
            # client：不连接外部服务的 Qdrant 内存实例
            client = QdrantClient(location=":memory:")
            # index：使用真实 Qdrant API 的长期记忆索引
            index = QdrantMemoryIndex(
                collection_name="test_memories",
                client=client,
            )
            try:
                index.upsert(memory, [1.0, 0.0, 0.0])

                allowed = index.search(
                    [1.0, 0.0, 0.0],
                    "tenant-1",
                    "user-1",
                )
                blocked = index.search(
                    [1.0, 0.0, 0.0],
                    "tenant-1",
                    "user-2",
                )
            finally:
                client.close()

        self.assertEqual([memory.id], [hit.memory_id for hit in allowed])
        self.assertEqual([], blocked)


if __name__ == "__main__":
    unittest.main()
