from typing import Any, Callable, Dict, List

from mini_agemem.memory_tools import (
    MemoryTools,
    ToolResult,
)

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",#! 告诉模型这是一个函数工具
        "function": {
            "name":"add_memory",
            "description": "添加一条新的长期记忆。",#! 告诉模型什么时候应该使用它
            "parameters":{ #! 描述工具参数
                "type":"object",
                "properties":{
                    "content":{
                        "type": "string",
                        "description": "需要保存的记忆内容。",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "记忆的分类信息。",
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "记忆类型",
                        "default": "general",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_memory",
            "description": "检索与查询相关的长期记忆。",
            "parameters":{
                "type": "object",
                "properties":{
                    "query":{
                        "type": "string",
                        "description": "用于检索记忆的查询文本。",
                    },
                    "top_k":{
                        "type": "integer",
                        "description": "最多返回多少条记忆。",
                        "default": 3,
                    },
                    "metadata_filter": {
                        "type":"object",
                        "description": "可选的元数据过滤条件。"
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function":{
            "name": "update_memory",
            "description": "更新一条已有长期记忆。",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id":{
                        "type": "string",
                        "description": "要更新的记忆ID。",
                    },
                    "content":{
                        "type": "string",
                        "description":"更新后的记忆内容。",
                    },
                    "metadata":{
                        "type":"object",
                        "description":"需要合并的元数据。",
                    },
                },
                "required": ["memory_id"],
            },
        },
    },
    {
        "type": "function",
        "function":{
            "name": "delete_memory",
            "description": "永久删除一条长期记忆。",
            "parameters":{
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "要删除的记忆ID。",
                    },
                    "confirmation": {
                        "type": "boolean",
                        "description": "必须为True才执行删除。",
                    },
                },
                "required": [
                    "memory_id",
                    "confirmation",
                ],
            },
        },
    },
]


#* ...表示参数形式暂时不限定
#* Callable 可以被调用的对象, 例如函数或方法
#* ToolResult 调用后返回ToolResult 
#TODO 第一个参数表示调用参数形式不固定， 第二个参数表示要返回一个ToolResult
ToolFunction = Callable[..., ToolResult]

class ToolRegistry:
    """保存工具说明书并分发工具调用。"""

    def __init__(
            self,
            memory_tools: MemoryTools,
    ) -> None:
        self._functions: Dict[
            str,
            ToolFunction,
        ] = {
            "add_memory": (
                memory_tools.add_memory
            ),
            "retrieve_memory": (
                memory_tools.retrieve_memory
            ),
            "update_memory": (
                memory_tools.update_memory
            ),
            "delete_memory": (
                memory_tools.delete_memory
            ),
        }

    def get_schemas(
            self,
    ) -> List[Dict[str, Any]]:
        return TOOL_SCHEMAS

    def call(
            self,
            name: str,
            arguments: Dict[str, Any],
    ) -> ToolResult:
        #! 这里返回的是name对应的MemoryTool
        function = self._functions.get(name)

        if function is None:
            return ToolResult(
                success = False,
                text = f"不存在工具: {name}",
            )

        return function(**arguments)