import threading

from qdrant_client import QdrantClient, models

from agent.memory.models import LongTermMemory, MemorySearchHit, MemoryType


class MemoryIndexConfigurationError(RuntimeError):
    pass


class QdrantMemoryIndex:
    # 初始化只保存向量与检索过滤字段的 Qdrant 索引客户端
    # url：Qdrant HTTP 服务地址
    # api_key：Qdrant 访问密钥，本地无鉴权服务可以为空
    # collection_name：长期记忆 Collection 名称
    # client：测试或扩展时注入的 QdrantClient
    # timeout：单次 Qdrant 请求超时秒数
    def __init__(
        self,
        url="http://localhost:6333",
        api_key=None,
        collection_name="agent_memories",
        client=None,
        timeout=10,
    ):
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name 不能为空")
        self.collection_name = collection_name.strip()
        self.client = client or QdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout,
        )
        # collection_lock：防止并行任务重复创建 Collection
        self.collection_lock = threading.Lock()
        # vector_size：已经确认的 Collection 向量维度
        self.vector_size = None

    # 释放 Qdrant Client 持有的网络连接
    def close(self):
        self.client.close()

    # 获取现有 Collection 使用的向量维度
    def _existing_vector_size(self):
        # collection：Qdrant 返回的 Collection 配置信息
        collection = self.client.get_collection(self.collection_name)
        # vectors：Collection 使用的向量参数
        vectors = collection.config.params.vectors
        if isinstance(vectors, models.VectorParams):
            return vectors.size
        raise MemoryIndexConfigurationError("长期记忆 Collection 不支持命名向量配置")

    # 确保 Collection 和用户隔离字段索引已经创建
    # vector_size：Embedding 模型实际生成的向量维度
    def ensure_collection(self, vector_size):
        if type(vector_size) is not int or vector_size < 1:
            raise ValueError("vector_size 必须是正整数")
        if self.vector_size is not None:
            if self.vector_size != vector_size:
                raise MemoryIndexConfigurationError(
                    f"向量维度不一致：Collection={self.vector_size}，输入={vector_size}"
                )
            return

        with self.collection_lock:
            if self.vector_size is not None:
                if self.vector_size != vector_size:
                    raise MemoryIndexConfigurationError(
                        f"向量维度不一致：Collection={self.vector_size}，输入={vector_size}"
                    )
                return

            if self.client.collection_exists(self.collection_name):
                # existing_size：现有 Collection 配置的向量维度
                existing_size = self._existing_vector_size()
                if existing_size != vector_size:
                    raise MemoryIndexConfigurationError(
                        f"Collection 向量维度为 {existing_size}，当前模型输出 {vector_size}"
                    )
            else:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )

            # collection：用于检查哪些 Payload 字段还没有建立索引
            collection = self.client.get_collection(self.collection_name)
            # indexed_fields：已经存在 Payload 索引的字段集合
            indexed_fields = set(collection.payload_schema)
            # payload_indexes：用户隔离和常用过滤字段的索引类型
            payload_indexes = {
                "tenant_id": models.PayloadSchemaType.KEYWORD,
                "user_id": models.PayloadSchemaType.KEYWORD,
                "project_id": models.PayloadSchemaType.KEYWORD,
                "memory_type": models.PayloadSchemaType.KEYWORD,
                "status": models.PayloadSchemaType.KEYWORD,
                "version": models.PayloadSchemaType.INTEGER,
                "expires_at": models.PayloadSchemaType.DATETIME,
            }
            for field_name, field_schema in payload_indexes.items():
                if field_name not in indexed_fields:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=field_schema,
                        wait=True,
                    )
            self.vector_size = vector_size

    # 使用 memory_id 幂等写入或覆盖一条记忆向量
    # memory：SQLite 中的最新完整记忆
    # vector：根据 memory.content 生成的向量
    def upsert(self, memory, vector):
        if not isinstance(memory, LongTermMemory):
            raise TypeError("memory 必须是 LongTermMemory")
        if not vector:
            raise ValueError("vector 不能为空")
        self.ensure_collection(len(vector))
        # payload：Qdrant 用于隔离和过滤的最小结构化字段
        payload = {
            "tenant_id": memory.tenant_id,
            "user_id": memory.user_id,
            "project_id": memory.project_id or "",
            "memory_type": memory.memory_type.value,
            "status": memory.status.value,
            "importance": memory.importance,
            "confidence": memory.confidence,
            "version": memory.version,
            "created_at": memory.created_at.isoformat(),
            "expires_at": (
                memory.expires_at.isoformat() if memory.expires_at is not None else None
            ),
        }
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=memory.id,
                    vector=vector,
                    payload=payload,
                )
            ],
            wait=True,
        )

    # 幂等删除指定记忆的 Qdrant Point
    # memory_id：SQLite 与 Qdrant 共用的记忆标识
    def delete(self, memory_id):
        # Collection 尚未创建意味着目标 Point 一定不存在，删除可直接视为成功
        if not self.client.collection_exists(self.collection_name):
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=[memory_id]),
            wait=True,
        )

    # 在可信用户边界内执行语义检索
    # vector：当前问题生成的查询向量
    # tenant_id：经过身份认证的租户标识
    # user_id：经过身份认证的用户标识
    # project_id：需要限定的项目标识，None 表示不限制项目
    # memory_types：允许召回的记忆类型
    # limit：最多返回的记忆数量
    # score_threshold：最低语义相关度
    def search(
        self,
        vector,
        tenant_id,
        user_id,
        project_id=None,
        memory_types=None,
        limit=8,
        score_threshold=None,
    ):
        if not vector:
            raise ValueError("vector 不能为空")
        if not tenant_id or not user_id:
            raise ValueError("tenant_id 和 user_id 不能为空")
        self.ensure_collection(len(vector))
        # conditions：所有查询都必须包含的租户和用户隔离条件
        conditions = [
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=tenant_id),
            ),
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            ),
            models.FieldCondition(
                key="status",
                match=models.MatchValue(value="active"),
            ),
        ]
        if project_id is not None:
            conditions.append(
                models.FieldCondition(
                    key="project_id",
                    # 当前项目同时允许召回用户级的全局长期记忆
                    match=models.MatchAny(any=[project_id, ""]),
                )
            )
        else:
            conditions.append(
                models.FieldCondition(
                    key="project_id",
                    match=models.MatchValue(value=""),
                )
            )
        if memory_types:
            # type_values：Qdrant MatchAny 使用的记忆类型文本
            type_values = [
                item.value if isinstance(item, MemoryType) else str(item)
                for item in memory_types
            ]
            conditions.append(
                models.FieldCondition(
                    key="memory_type",
                    match=models.MatchAny(any=type_values),
                )
            )

        # response：Qdrant 返回的相关向量点
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=models.Filter(must=conditions),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=False,
            with_vectors=False,
        )
        return [
            MemorySearchHit(memory_id=str(point.id), score=point.score)
            for point in response.points
        ]

