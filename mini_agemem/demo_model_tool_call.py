import dashscope
import asyncio
import os

from dotenv import load_dotenv, find_dotenv

#TODO 将AgentScope内部的消息格式转换成模型接口能接受的格式
from agentscope.formatter import DashScopeChatFormatter

#TODO 将远程的Qwen模型包装成一个可以被AgentScope调用的模型对象
from agentscope.model import DashScopeChatModel
#TODO 负责管理工具， 是一个工具注册和调度系统
#* 1. 将工具函数注册到Toolkit
#* 2. 获取这些工具对应的JSON Schema
#* 3. 接收大模型的调用
#* 4. 通过Toolkit执行对应的函数
from agentscope.tool import Toolkit

from mini_agemem.agentscope_memory_tools import(
    AgentScopeMemoryTools,
)
from mini_agemem.embedding import FakeEmbeddingClient
from mini_agemem.memory_manager import MemoryManager
from mini_agemem.memory_tools import MemoryTools
from mini_agemem.simple_agent import (
    SimpleMemoryAgent,
)

SYSTEM_PROMPT = """
你是一个具有长期记忆工具的助手。

当用户明确要求“请记住”某项信息时，
你应该调用 add_memory。

当用户要求回忆、查找过去的信息时，
你应该调用 retrieve_memory。

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
"""

_ = load_dotenv(find_dotenv())

async def main() -> None:
    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "没有找到 DASHSCOPE_API_KEY",
        )

    model_name = (
        os.getenv("AGENT_MODEL_NAME")
        or "qwen-max"
    )

    base_http_api_url = os.getenv(
        "DASHSCOPE_BASE_HTTP_API_URL",
    )

    #! stream：是否使用流式输出， 流式输出就是一个Token一个Token的返回内容
    model = DashScopeChatModel(
        model_name = model_name,
        api_key=api_key,
        stream = False,
        enable_thinking=False,
        base_http_api_url = base_http_api_url,
    )

    #! 创建消息格式转换器
    formatter = DashScopeChatFormatter()

    manager = MemoryManager(
        embedding_client = FakeEmbeddingClient(),
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

    first_answer = await agent.reply(
        "请记住: "
        "我最喜欢使用Python编程。"
    )

    print(
        "\n第一轮最终回答 =",
        first_answer,
    )

    print(
        "保存后记忆数量 =",
        manager.size(),
    )

    second_answer = await agent.reply(
       "请回忆一下:"
       "我最喜欢使用什么编程语言？"
    )

    print(
        "\n第二轮最终回答 =",
        second_answer
    )

    print(
        "检索后的记忆数量",
        manager.size(),
    )



async def run() -> None:
    try:
        await main()
    finally:
        await dashscope.close_shared_aio_session()

        # 给 Windows 上的 HTTPS 连接一点时间完成关闭
        await asyncio.sleep(0.25)


if __name__ == "__main__":
    asyncio.run(run())