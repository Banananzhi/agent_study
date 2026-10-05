import math
from typing import Protocol

import httpx


class EmbeddingProvider(Protocol):
    # model_name：生成向量时使用的模型名称
    model_name: str

    # 批量生成适合文档入库的向量
    # texts：需要生成向量的文本列表
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    # 生成适合语义查询的单条向量
    # text：需要查询相关记忆的文本
    def embed_query(self, text: str) -> list[float]:
        ...


class OpenAICompatibleEmbeddingProvider:
    # 初始化通过 OpenAI 兼容接口调用的在线 Embedding 组件
    # model_name：在线服务提供的 Embedding 模型名称
    # base_url：OpenAI 兼容 API 根地址
    # api_key：在线 Embedding 服务访问密钥
    # timeout：单次向量请求超时秒数
    # dimensions：可选的目标向量维度
    # client：测试或扩展时注入的 httpx.Client
    def __init__(
        self,
        model_name="qwen3.7-text-embedding-flash",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=None,
        timeout=30,
        dimensions=None,
        client=None,
    ):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name 不能为空")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url 不能为空")
        if type(timeout) not in {int, float} or timeout <= 0:
            raise ValueError("timeout 必须大于 0")
        if dimensions is not None and (type(dimensions) is not int or dimensions < 1):
            raise ValueError("dimensions 必须是正整数或 None")
        self.model_name = model_name.strip()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.dimensions = dimensions
        # owns_client：记录 HTTP Client 是否由当前 Provider 创建
        self.owns_client = client is None
        # client：复用连接池的同步 HTTP Client
        self.client = client or httpx.Client(timeout=timeout)

    # 释放当前 Provider 自己创建的 HTTP 连接池
    def close(self):
        if self.owns_client:
            self.client.close()

    # 校验输入文本并调用在线 Embedding 接口
    # texts：需要转换成向量的有序文本列表
    def _embed(self, texts):
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts 必须是非空文本列表")
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise RuntimeError(
                "请在 .env 中设置 AGENT_EMBEDDING_API_KEY 或 DASHSCOPE_API_KEY"
            )
        # request_body：OpenAI 兼容 Embedding 请求参数
        request_body = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }
        if self.dimensions is not None:
            request_body["dimensions"] = self.dimensions
        # response：在线 Embedding 服务返回的 HTTP 响应
        response = self.client.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key.strip()}",
                "Content-Type": "application/json",
            },
            json=request_body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        # payload：在线服务返回的标准 Embedding JSON
        payload = response.json()
        # data：按原始输入下标排序的向量记录
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            raise ValueError("Embedding 服务返回的 data 数量与输入不一致")
        try:
            # ordered_data：防止服务端并发处理后打乱输入顺序
            ordered_data = sorted(data, key=lambda item: item["index"])
            # vectors：从标准响应中提取的浮点向量列表
            vectors = [item["embedding"] for item in ordered_data]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Embedding 服务返回格式不合法") from error
        # vector_size：第一条响应确定的实际向量维度
        vector_size = len(vectors[0]) if vectors and isinstance(vectors[0], list) else 0
        if vector_size < 1:
            raise ValueError("Embedding 服务返回了空向量")
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != vector_size:
                raise ValueError("Embedding 服务返回的向量维度不一致")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in vector
            ):
                raise ValueError("Embedding 服务返回了非法向量值")
        return [[float(value) for value in vector] for vector in vectors]

    # 批量生成适合长期记忆入库的向量
    # texts：需要生成向量的文本列表
    def embed_documents(self, texts):
        return self._embed(texts)

    # 生成适合长期记忆语义查询的单条向量
    # text：当前任务或问题文本
    def embed_query(self, text):
        return self._embed([text])[0]
