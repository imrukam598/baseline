from mini_agemem.embedding import (
    FakeEmbeddingClient,
)
from mini_agemem.memory_manager import (
    MemoryManager,
)
from mini_agemem.memory_tools import MemoryTools
from mini_agemem.tool_registry import (
    ToolRegistry,
)


def test_tool_registry() -> None:
    manager = MemoryManager(
        embedding_client=FakeEmbeddingClient(),
    )

    memory_tools = MemoryTools(
        memory_manager=manager,
    )

    registry = ToolRegistry(
        memory_tools=memory_tools,
    )

    tool_names = [
        schema["function"]["name"]
        for schema in registry.get_schemas()
    ]

    print("tool_names =", tool_names)

    assert tool_names == [
        "add_memory",
        "retrieve_memory",
        "update_memory",
        "delete_memory",
    ]

    add_call = {
        "name": "add_memory",
        "arguments": {
            "content": "用户喜欢 Python 编程",
            "memory_type": "preference",
        },
    }

    add_result = registry.call(
        name=add_call["name"],
        arguments=add_call["arguments"],
    )

    print(
        "add_result =",
        add_result.to_dict(),
    )

    assert add_result.success is True

    memory_id = add_result.data["memory_id"]

    retrieve_result = registry.call(
        name="retrieve_memory",
        arguments={
            "query": "Python 编程",
            "top_k": 1,
        },
    )

    print(
        "retrieve_result =",
        retrieve_result.to_dict(),
    )

    retrieved_id = (
        retrieve_result
        .data["memories"][0]["memory_id"]
    )

    assert retrieved_id == memory_id

    update_result = registry.call(
        name="update_memory",
        arguments={
            "memory_id": memory_id,
            "content": "用户正在学习 Rust",
        },
    )

    assert update_result.success is True

    updated_item = manager.get_memory(
        memory_id
    )

    assert updated_item is not None
    assert updated_item.content == (
        "用户正在学习 Rust"
    )

    delete_result = registry.call(
        name="delete_memory",
        arguments={
            "memory_id": memory_id,
            "confirmation": True,
        },
    )

    print(
        "delete_result =",
        delete_result.to_dict(),
    )

    assert delete_result.success is True
    assert manager.size() == 0

    print("ToolRegistry主流程测试通过")


if __name__ == "__main__":
    test_tool_registry()