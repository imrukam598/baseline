from abc import ABC, abstractmethod
from typing import List, Optional

import os


class EmbeddingClient(ABC):
    """所有文本向量模型都应遵守的接口"""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """把一段文本转换为浮点数向量."""

        raise NotImplementedError

class FakeEmbeddingClient(EmbeddingClient):
    """用于离线测试的关键词向量模型"""

    def __init__(self) -> None:
        self.keyword_groups = [
            ["python", "rust", "代码", "编程", "程序"],
            ["上海", "旅行", "旅游", "酒店", "景点"],
            ["咖啡", "食物", "餐厅", "美食", "吃饭"],
        ]

    def embed(self, text: str) -> List[float]:

        normalized_text = text.lower()

        category_vector = []

        for keyword_group in self.keyword_groups:
            keyword_count = sum(
                1.0
                for keyword in keyword_group
                if keyword.lower() in normalized_text
            )

            category_vector.append(keyword_count)

        has_know_category = any(
            value > 0.0
            for value in category_vector
        )

        if has_know_category:
            unknown_dimension = 0.0
        else:
            unknown_dimension = 1.0

        return category_vector + [unknown_dimension]

class DashScopeEmbeddingClient(EmbeddingClient):
    """"调用DashScope text-embedding-v4的真实向量客户端"""

    def __init__(
            self,
            api_key: Optional[str] = None,
            model_name: str = "text-embedding-v4",
            dimensions: int = 256,
            base_url: Optional[str] = None,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions 必须大于0")

        resolved_api_key = (
            api_key
            or os.getenv("DASHSCOPE_API_KEY")
        )

        if not resolved_api_key:
            raise ValueError(
                "没有找到DashScope API Key."
                "请设置环境变量DASHSCOPE_API_KEY."
            )

        resolved_base_url = (
            base_url
            or os.getenv("OPENAI_URL")
        )

        if not resolved_base_url:
            raise ValueError(
                "没有找到 DashScope "
                "OpenAI 兼容 Base URL。"
            )

        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "缺少openai包,请先运行:"
                "python -m pip install 'openai>=1,<2'"
            ) from error

        self.model_name = model_name
        self.dimensions = dimensions

        self._client = OpenAI(
            api_key = resolved_api_key,
            base_url = resolved_base_url,
        )

    def embed(self, text: str) -> List[float]:
        if not text.strip():
            raise ValueError("需要向量化的文本不能为空")

        response = self._client.embeddings.create(
            model = self.model_name,
            input = text,
            dimensions = self.dimensions,
            encoding_format = "float",
        )

        if not response.data:
            raise RuntimeError(
                "DashScope没有返回embedding数据"
            )

        embedding = response.data[0].embedding

        if not embedding:
            raise RuntimeError(
                "DashScope返回了空embedding"
            )

        if len(embedding) != self.dimensions:
            raise RuntimeError(
                "embedding维度不符合预期:"
                f"预期 {self.dimensions}"
                f"实际{len(embedding)}"
            )

        return [
            float(value)
            for value in embedding
        ]