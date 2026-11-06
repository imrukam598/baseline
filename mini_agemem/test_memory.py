from mini_agemem.memory import InMemoryVectorStore, MemoryItem

def test_memory_item_initial_timestamps() -> None:
    item = MemoryItem(
        memory_id = "new-1",
        content = "新记忆",
    )
    
    assert item.created_at
    assert item.updated_at
    assert (
        item.created_at
        == item.updated_at
    )

    print("MemoryItem初始时间测试通过")

def test_vector_store() -> None:

    store = InMemoryVectorStore()

    python_memory = MemoryItem(
        memory_id="m1",
        content="用户喜欢 Python",
        metadata={"topic": "programming"},
        embedding=[1.0, 0.0],
    )

    rust_memory = MemoryItem(
        memory_id="m2",
        content="用户正在学习 Rust",
        metadata={"topic": "programming"},
        embedding=[0.8, 0.2],
    )

    travel_memory = MemoryItem(
        memory_id="m3",
        content="用户计划去上海旅行",
        metadata={"topic": "travel"},
        embedding=[0.0, 1.0],
    )

    store.add(python_memory)
    store.add(rust_memory)
    store.add(travel_memory)

    actual_size = store.size()
    expected_size = 3

    print("actual_size =", actual_size)
    print("expected_size =", expected_size)

    assert actual_size == expected_size

    results = store.search(
        query_embedding=[1.0, 0.0],
        top_k=2,
    )

    actual_ids = [item.memory_id for item, score in results]
    expected_ids = ["m1", "m2"]

    print("actual_ids =", actual_ids)
    print("expected_ids =", expected_ids)

    assert actual_ids == expected_ids

    updated = store.update(
        memory_id="m1",
        new_content="用户最喜欢的语言是 Python",
        new_metadata={"status": "updated"},
    )

    assert updated is True

    updated_item = store.get("m1")

    assert updated_item is not None
    assert updated_item.content == "用户最喜欢的语言是 Python"
    assert updated_item.metadata == {
        "topic": "programming",
        "status": "updated",
    }

    deleted = store.delete("m3")

    assert deleted is True
    assert store.get("m3") is None
    assert store.size() == 2

    print("所有测试通过")


if __name__ == "__main__":
    test_memory_item_initial_timestamps()
    test_vector_store()