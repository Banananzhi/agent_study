import json
import tempfile
import unittest
from pathlib import Path

import httpx
from langchain_core.messages import AIMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from qdrant_client import QdrantClient

from agent import Agent
from agent.memory.extractor import StructuredMemoryExtractor
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
                                "name": "MemoryExtractionBatch",
                                "arguments": MemoryExtractionBatch().model_dump_json(),
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

        recalled = self.service.recall("我喜欢怎样的解释", "tenant-1", "user-1")

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
                    )
                ]
            )
        )
        # agent：使用固定回答避免测试访问真实模型
        agent = MemoryScriptedAgent(
            [AIMessage(content="已经确认长期记忆方案。", id="assistant-final")],
            memory_service=self.service,
            memory_extractor=extractor,
            tenant_id="tenant-1",
            user_id="user-1",
            project_id="agent-study",
        )

        answer = agent.run("继续完善长期记忆")

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
