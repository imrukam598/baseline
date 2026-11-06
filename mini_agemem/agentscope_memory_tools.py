"""
转换为AgentScope的ToolResponse
"""
from typing import Dict, Optional
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

from mini_agemem.memory_tools import (
    MemoryTools,
    ToolResult,
)

#* Toolkit 是工具注册器和分发器。
#* get_json_schemas() 生成给大模型看的工具说明书。
#* ToolUseBlock 描述“大模型想调用哪个工具、传什么参数”。
#* ToolResponse 是 AgentScope 规定的工具返回格式。
#* AgentScopeMemoryTools 只是把你的 ToolResult 转成 ToolResponse。

def _to_tool_response(
        result:ToolResult,
) -> ToolResponse:
    """把项目自己的ToolResult转成AgentScope ToolResponse。"""

    return ToolResponse(
        content = [
            TextBlock(
                type = "text",
                text = result.text,
            ),
        ],
        metadata = result.to_dict(),
    )



class AgentScopeMemoryTools:
    """把我们自己的MemoryTools转换成AgentScope工具。"""

    def __init__(
            self,
            memory_tools: MemoryTools,
    ) -> None:
        self._memory_tools = memory_tools

    async def add_memory(
            self,
            content: str,
            metadata: Optional[Dict[str, str]] = None,
            memory_type: str = "general",
    ) -> ToolResponse:
        """添加一条新的长期记忆。
        
        Args:
            content (`str`):
                需要保存的记忆内容。
            metadata (`dict`, defaults to `None`):
                可选的记忆分类信息。
            memory_type(`str`, defaults to `"general"`):
                记忆类型。
        """

        result = self._memory_tools.add_memory(
            content = content,
            metadata = metadata,
            memory_type = memory_type,
        )

        return _to_tool_response(result)

    async def retrieve_memory(
            self,
            query: str,
            top_k: int = 3,
            metadata_filter: Optional[
                Dict[str, str]
            ] = None
    ) -> ToolResponse:
        """检索与查询相关的长期记忆。
        
        Args:
            query (`str`):
                用来检索记忆的查询文本。
            top_k (`int`, defaults to `3`):
                最多返回多少条记忆。
            metadata_filter (`dict`, defaults to `None`):
                只用于匹配 MemoryItem.metadata
                中的字段，例如 type 和 stage。
                不能用于匹配 memory_id、
                created_at 或 updated_at。
        """

        result = self._memory_tools.retrieve_memory(
            query = query,
            top_k = top_k,
            metadata_filter = metadata_filter,
        )

        return _to_tool_response(result)

    async def update_memory(
            self,
            memory_id: str,
            content: Optional[str] = None,
            metadata: Optional[Dict[str, str]] =None,
    ) -> ToolResponse:
        """更新已有长期记忆。
        
        Args:
            memory_id (`str`):
                需要更新的记忆编号。
            content (`str`, defaults tp `None`):
                更新后的记忆内容。
            metadata(`dict`, defaults to `None`):
                需要合并的元数据。
        """

        result = self._memory_tools.update_memory(
            memory_id = memory_id,
            content = content,
            metadata = metadata,
        )

        return _to_tool_response(result)

    async def delete_memory(
            self,
            memory_id: str,
            confirmation: bool = False,
    ) -> ToolResponse:
        """永久删除一条长期记忆。
        
        Args:
            memory_id (`str`):
                需要删除的记忆编号。
            confirmation (`bool`, defaults to `False`):
                必须为True才能执行删除。
        """

        result = self._memory_tools.delete_memory(
            memory_id = memory_id,
            confirmation = confirmation,
        )

        return _to_tool_response(result)







    

