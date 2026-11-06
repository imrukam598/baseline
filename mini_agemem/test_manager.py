from mini_agemem.embedding import FakeEmbeddingClient
from mini_agemem.memory_manager import MemoryManager



def test_memory_manager() -> None:
    embedding_client = FakeEmbeddingClient()

    manager = MemoryManager(
        embedding_client=embedding_client,
    )

    manager.add_memory(
        memory_id="m1",
        content="用户喜欢 Python",
        metadata={"topic": "programming"},
    )

    manager.add_memory(
        memory_id="m2",
        content="用户正在学习 Rust",
        metadata={
            "topic": "programming",
            "level": "beginner",
        },
    )

    manager.add_memory(
        memory_id="m3",
        content="用户计划去上海旅行",
        metadata={"topic": "travel"},
    )

    actual_size = manager.size()
    expected_size = 3

    print("actual_size =", actual_size)
    print("expected_size =", expected_size)

    assert actual_size == expected_size

    programming_results = manager.retrieve(
        query="我应该学习什么编程语言？",
        top_k=2,
    )

    actual_programming_ids = [
        item.memory_id
        for item in programming_results
    ]

    expected_programming_ids = ["m1", "m2"]

    print(
        "actual_programming_ids =",
        actual_programming_ids,
    )
    print(
        "expected_programming_ids =",
        expected_programming_ids,
    )

    assert actual_programming_ids == expected_programming_ids

    filtered_results = manager.retrieve(
        query="编程学习",
        top_k=5,
        metadata_filter={"level": "beginner"},
    )

    actual_filtered_ids = [
        item.memory_id
        for item in filtered_results
    ]

    expected_filtered_ids = ["m2"]

    print("actual_filtered_ids =", actual_filtered_ids)
    print("expected_filtered_ids =", expected_filtered_ids)

    assert actual_filtered_ids == expected_filtered_ids

    updated = manager.update_memory(
        memory_id="m3",
        content="用户最喜欢喝咖啡",
        metadata={"topic": "food"},
    )

    assert updated is True

    travel_results = manager.retrieve(
        query="上海旅行",
    )

    actual_travel_ids = [
        item.memory_id
        for item in travel_results
    ]

    expected_travel_ids = []

    print("actual_travel_ids =", actual_travel_ids)
    print("expected_travel_ids =", expected_travel_ids)

    assert actual_travel_ids == expected_travel_ids

    food_results = manager.retrieve(
        query="用户喜欢什么食物或饮品？",
    )

    actual_food_ids = [
        item.memory_id
        for item in food_results
    ]

    expected_food_ids = ["m3"]

    print("actual_food_ids =", actual_food_ids)
    print("expected_food_ids =", expected_food_ids)

    assert actual_food_ids == expected_food_ids

    deleted = manager.delete_memory("m1")

    assert deleted is True
    assert manager.get_memory("m1") is None
    assert manager.size() == 2

    print("MemoryManager 所有测试通过")


if __name__ == "__main__":
    test_memory_manager()