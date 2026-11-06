import math
import os
from dotenv import load_dotenv
from os import getenv

from mini_agemem.embedding import (
    DashScopeEmbeddingClient,
)
from mini_agemem.memory_manager import MemoryManager

_ = load_dotenv()

def test_dashscope_embedding() -> None:
    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "请先设置 DASHSCOPE_API_KEY"
        )

    embedding_client = DashScopeEmbeddingClient(
        dimensions=256,
    )

    manager = MemoryManager(
        embedding_client=embedding_client,
    )

    added_item = manager.add_memory(
        memory_id="online-1",
        content="用户喜欢使用 Python 编写程序",
        metadata={"topic": "programming"},
    )

    actual_dimension = len(
        added_item.embedding or []
    )
    expected_dimension = 256

    print(
        "actual_dimension =",
        actual_dimension,
    )
    print(
        "expected_dimension =",
        expected_dimension,
    )

    assert actual_dimension == expected_dimension

    assert added_item.embedding is not None

    all_values_are_finite = all(
        math.isfinite(value)
        for value in added_item.embedding
    )

    print(
        "all_values_are_finite =",
        all_values_are_finite,
    )

    assert all_values_are_finite is True

    print(
        "first_five_values =",
        added_item.embedding[:5],
    )

    retrieved_items = manager.retrieve(
        query="用户喜欢使用 Python 编写程序",
        top_k=1,
    )

    actual_ids = [
        item.memory_id
        for item in retrieved_items
    ]
    expected_ids = ["online-1"]

    print("actual_ids =", actual_ids)
    print("expected_ids =", expected_ids)

    assert actual_ids == expected_ids

    print("DashScope在线测试通过")


if __name__ == "__main__":
    test_dashscope_embedding()