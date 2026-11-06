from pathlib import Path
import asyncio
import os

import dashscope
from dotenv import find_dotenv, load_dotenv

from agentscope.formatter import(
    DashScopeChatFormatter,
)
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit

from mini_agemem.agentscope_memory_tools import (
    AgentScopeMemoryTools,
)
from mini_agemem.embedding import DashScopeEmbeddingClient
from mini_agemem.memory_manager import MemoryManager
from mini_agemem.memory_tools import MemoryTools
from mini_agemem.simple_agent import SimpleMemoryAgent
from mini_agemem.sqlite_memory_store import (
    SQLiteMemoryStore,
)

SYSTEM_PROMPT = """
你是一个具有长期记忆工具的助手。

当用户明确要求记住一项信息时，
不要立即调用 add_memory。

首先调用 retrieve_memory,
使用需要保存的新信息作为 query,
检查长期记忆中是否已有相关事实。

检索后按照以下规则处理：

1. 如果没有相关记忆，调用 add_memory。

2. 如果已有记忆表达的是同一个事实，
   不要重复添加，直接告诉用户已经记住。

3. 如果新信息取代了同一对象的旧事实，
   使用检索结果中的 Memory ID
   调用 update_memory。

4. 如果新信息与已有记忆可以同时成立，
   调用 add_memory 保存为另一条记忆。

一次保存流程中不要重复调用 retrieve_memory。

当用户要求回忆、查找过去的信息时，
你应该调用 retrieve_memory。
retrieve_memory 的 metadata_filter
只能过滤 metadata 中的 type、stage 等字段。

不要把 memory_id、created_at 或 updated_at
放入 metadata_filter。

当用户要求修改已有记忆时，
你应该调用update_memory。

只有用户明确确认永久删除时，
才能调用delete_memory。

工具成功执行后，请根据工具结果回答用户，
不要重复执行同一个动作。

当用户询问已经保存的信息时,
即使该信息仍然出现在当前对话中,
也必须先调用retrieve_memory,
再根据检索结果回答用户。

当对话已经很长，或者用户明确要求压缩对话时，
调用 summary_context。

需要压缩全部旧消息时, span 使用 "all";
只压缩最近几条消息时, span 使用正整数字符串。

当用户要求从当前对话中删除无关、过时或不再需要的内容时，
调用 filter_context。

filter_context 的 criteria 应描述需要删除的内容，
而不是需要保留的内容。
"""

_ = load_dotenv(find_dotenv())

async def main() -> None:
    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "没有找到DASHSCOPE_API_KEY",
        )

    model_name = (
        os.getenv("AGENT_MODEL_NAME")
        or "qwen-max"
    )

    base_http_api_url = os.getenv(
        "DASHSCOPE_BASE_HTTP_API_URL",
    )

    embedding_base_url = os.getenv(
        "OPENAI_URL",
    )

    if not embedding_base_url:
        raise RuntimeError(
            "没有找到OPENAI_URL"
        )

    model = DashScopeChatModel(
        model_name = model_name,
        api_key = api_key,
        stream = False,
        enable_thinking = False,
        base_http_api_url = base_http_api_url,
    )

    formatter = DashScopeChatFormatter()

    database_path = (
        Path(__file__).resolve().parent
        /"data"
        /"memories.db"
    )

    store = SQLiteMemoryStore(
        database_path = database_path,
    )

    embedding_client = (
        DashScopeEmbeddingClient(
            api_key=api_key,
            model_name = (
                "text-embedding-v4"
            ),
            dimensions=256,
            base_url=(
                embedding_base_url
            ),
        )
    )

    manager = MemoryManager(
        embedding_client = (
            embedding_client
        ),
        store=store,
    )

    print(
        "已加载长期记忆数量 =",
        manager.size(),
    )

    memory_tools = MemoryTools(
        memory_manager = manager,
    )

    agentscope_tools = AgentScopeMemoryTools(
        memory_tools = memory_tools,
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


    agent = SimpleMemoryAgent(
        name="MiniAgeMem",
        system_prompt=SYSTEM_PROMPT,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        max_steps=4,
    )

    toolkit.register_tool_function(
        agent.summary_context,
    )
    toolkit.register_tool_function(
        agent.filter_context,
    )

    print("=== MiniAgeMem ===")
    print("输入 exit 或 quit 结束程序。\n")

    while True:
        try:
            user_text = input("[你] >").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[MiniAgeMem]再见。")
            break

        if not user_text:
            continue

        if user_text.lower() in {
            "exit",
            "quit",
        }:
            print("再见。")
            break

        answer = await agent.reply(
            user_text,
        )

        print(
            "当前长期记忆数量 =",
            manager.size(),
        )

        print(
            f"\n[MiniAgeMem] > {answer}\n"
        )

async def run() -> None:
    try:
        await main()
    finally:
        await dashscope.close_shared_aio_session()

        await asyncio.sleep(0.25)

if __name__ == "__main__":
    asyncio.run(run())
    