import asyncio
import json
from typing import Any, Dict

from agentscope.message import ToolUseBlock
from agentscope.tool import Toolkit, ToolResponse

from mini_agemem.agentscope_memory_tools import (
    AgentScopeMemoryTools,
)
from mini_agemem.embedding import FakeEmbeddingClient
from mini_agemem.memory_manager import MemoryManager
from mini_agemem.memory_tools import MemoryTools


async def call_tool(
    toolkit: Toolkit,
    name: str,
    arguments: Dict[str, Any],
) -> ToolResponse:
    """构造一次工具调用，并取出最后一个响应。"""

    tool_call = ToolUseBlock(
        type="tool_use",
        id=f"call-{name}",
        name=name,
        input=arguments,
    )

    response_stream = await toolkit.call_tool_function(
        tool_call,
    )

    final_response = None

    async for response in response_stream:
        final_response = response

    assert final_response is not None

    return final_response


async def test_agentscope_toolkit() -> None:
    manager = MemoryManager(
        embedding_client=FakeEmbeddingClient(),
    )

    memory_tools = MemoryTools(
        memory_manager=manager,
    )

    agentscope_tools = AgentScopeMemoryTools(
        memory_tools=memory_tools,
    )

    toolkit = Toolkit()

    tool_functions = [
        agentscope_tools.add_memory,
        agentscope_tools.retrieve_memory,
        agentscope_tools.update_memory,
        agentscope_tools.delete_memory,
    ]

    for tool_function in tool_functions:
        toolkit.register_tool_function(
            tool_function,
        )

    schemas = toolkit.get_json_schemas()

    print(
        json.dumps(
            schemas,
            indent=2,
            ensure_ascii=False,
        ),
    )

    tool_names = [
        schema["function"]["name"]
        for schema in schemas
    ]

    assert tool_names == [
        "add_memory",
        "retrieve_memory",
        "update_memory",
        "delete_memory",
    ]

    add_response = await call_tool(
        toolkit=toolkit,
        name="add_memory",
        arguments={
            "content": "用户喜欢 Python 编程",
            "memory_type": "preference",
        },
    )

    memory_id = add_response.metadata[
        "data"
    ]["memory_id"]

    retrieve_response = await call_tool(
        toolkit=toolkit,
        name="retrieve_memory",
        arguments={
            "query": "Python 编程",
            "top_k": 1,
        },
    )

    retrieved_id = retrieve_response.metadata[
        "data"
    ]["memories"][0]["memory_id"]

    assert retrieved_id == memory_id

    update_response = await call_tool(
        toolkit=toolkit,
        name="update_memory",
        arguments={
            "memory_id": memory_id,
            "content": "用户正在学习 Rust",
        },
    )

    assert update_response.metadata["success"] is True

    delete_response = await call_tool(
        toolkit=toolkit,
        name="delete_memory",
        arguments={
            "memory_id": memory_id,
            "confirmation": True,
        },
    )

    assert delete_response.metadata["success"] is True
    assert manager.size() == 0

    print("四个 AgentScope 记忆工具主流程测试通过")


if __name__ == "__main__":
    asyncio.run(test_agentscope_toolkit())