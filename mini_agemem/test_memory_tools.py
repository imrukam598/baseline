import uuid

from mini_agemem.embedding import (
    FakeEmbeddingClient,
)
from mini_agemem.memory_manager import (
    MemoryManager,
)
from mini_agemem.memory_tools import MemoryTools


def test_memory_tools() -> None:
    embedding_client = FakeEmbeddingClient()

    manager = MemoryManager(
        embedding_client=embedding_client,
    )

    tools = MemoryTools(
        memory_manager=manager,
        current_stage=1,
    )

    add_programming_result = tools.add_memory(
        content="用户喜欢 Python 编程",
        metadata={"topic": "programming"},
        memory_type="preference",
    )

    print(
        "add_programming_result =",
        add_programming_result.to_dict(),
    )

    assert add_programming_result.success is True

    programming_id = (
        add_programming_result.data["memory_id"]
    )

    uuid.UUID(programming_id)

    stored_programming_item = (
        manager.get_memory(programming_id)
    )

    assert stored_programming_item is not None
    assert stored_programming_item.metadata == {
        "topic": "programming",
        "type": "preference",
        "stage": "1",
    }

    add_travel_result = tools.add_memory(
        content="用户计划去上海旅行",
        metadata={"topic": "travel"},
        memory_type="plan",
    )

    assert add_travel_result.success is True

    travel_id = add_travel_result.data[
        "memory_id"
    ]

    actual_size = manager.size()
    expected_size = 2

    print("actual_size =", actual_size)
    print("expected_size =", expected_size)

    assert actual_size == expected_size

    retrieve_result = tools.retrieve_memory(
        query="Python 编程",
        top_k=3,
        metadata_filter={
            "topic": "programming",
        },
    )

    print(
        "retrieve_result =",
        retrieve_result.to_dict(),
    )

    assert retrieve_result.success is True

    actual_retrieved_ids = [
        memory["memory_id"]
        for memory in retrieve_result.data[
            "memories"
        ]
    ]

    expected_retrieved_ids = [
        programming_id,
    ]

    print(
        "actual_retrieved_ids =",
        actual_retrieved_ids,
    )
    print(
        "expected_retrieved_ids =",
        expected_retrieved_ids,
    )

    assert (
        actual_retrieved_ids
        == expected_retrieved_ids
    )

    update_result = tools.update_memory(
        memory_id=travel_id,
        content="用户最喜欢喝咖啡",
        metadata={"topic": "food"},
    )

    print(
        "update_result =",
        update_result.to_dict(),
    )

    assert update_result.success is True

    food_result = tools.retrieve_memory(
        query="咖啡和食物",
        top_k=1,
    )

    actual_food_ids = [
        memory["memory_id"]
        for memory in food_result.data[
            "memories"
        ]
    ]

    expected_food_ids = [travel_id]

    print(
        "actual_food_ids =",
        actual_food_ids,
    )
    print(
        "expected_food_ids =",
        expected_food_ids,
    )

    assert actual_food_ids == expected_food_ids

    cancelled_delete_result = (
        tools.delete_memory(
            memory_id=programming_id,
        )
    )

    print(
        "cancelled_delete_result =",
        cancelled_delete_result.to_dict(),
    )

    assert (
        cancelled_delete_result.success
        is False
    )
    assert manager.size() == 2

    confirmed_delete_result = (
        tools.delete_memory(
            memory_id=programming_id,
            confirmation=True,
        )
    )

    print(
        "confirmed_delete_result =",
        confirmed_delete_result.to_dict(),
    )

    assert (
        confirmed_delete_result.success
        is True
    )
    assert manager.get_memory(
        programming_id
    ) is None
    assert manager.size() == 1

    missing_delete_result = (
        tools.delete_memory(
            memory_id="not-found",
            confirmation=True,
        )
    )

    assert missing_delete_result.success is False

    print("MemoryTools所有测试通过")


if __name__ == "__main__":
    test_memory_tools()