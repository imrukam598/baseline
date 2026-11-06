from mini_agemem.embedding import (
    FakeEmbeddingClient,
)
from mini_agemem.memory_manager import (
    MemoryManager,
)
from mini_agemem.memory_tools import (
    MemoryTools,
)


def test_memory_deduplication() -> None:
    manager = MemoryManager(
        embedding_client=(
            FakeEmbeddingClient()
        ),
    )

    tools = MemoryTools(
        memory_manager=manager,
    )

    first_result = tools.add_memory(
        content="用户喜欢 Python",
        memory_type="preference",
    )

    second_result = tools.add_memory(
        content="  用户喜欢   python  ",
        memory_type="preference",
    )

    assert first_result.success is True
    assert second_result.success is True

    assert manager.size() == 1

    assert (
        first_result.data["memory_id"]
        == second_result.data["memory_id"]
    )

    assert (
        second_result.data["duplicate"]
        is True
    )

    print(
        "长期记忆精确去重测试通过"
    )


if __name__ == "__main__":
    test_memory_deduplication()