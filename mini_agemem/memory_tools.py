"""
把操作变成工具语义
"""
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from mini_agemem.memory_manager import MemoryManager

@dataclass
class ToolResult:
    """一次工具调用的统一返回结果"""

    success: bool
    text: str
    data:Dict[str, Any] = field(default_factory = dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "text": self.text,
            "data": self.data,
        }

class MemoryTools:
    """提供给Agent使用的长期记忆工具"""

    def __init__(
            self,
            memory_manager: MemoryManager,
            current_stage: int = 0,
    ) -> None:
        self._memory_manager = memory_manager
        self.current_stage = current_stage

    def add_memory(
            self,
            content: str,
            metadata: Optional[Dict[str, str]] = None,
            memory_type: str = "general",
    ) -> ToolResult:
        """添加一条新的长期记忆"""
        if not content.strip():
            return ToolResult(
                success = False,
                text = "添加记忆失败:content不能为空。",
            )

        existing_item = (
            self._memory_manager
            .find_exact_memory(
                content
            )
        )

        if existing_item is not None:
            return ToolResult(
                success = True,
                text = (
                    "相同的长期记忆已经存在，"
                    "没有重复添加。"
                    f"Memory ID"
                    f"{existing_item.memory_id};"
                    f"内容："
                    f"{existing_item.content}"
                ),
                data = {
                    "memory_id":(
                        existing_item.memory_id
                    ),
                    "content":(
                        existing_item.content
                    ),
                    "metadata": dict(
                        existing_item.metadata
                    ),
                    "duplicate": True,
                    "created_at": (
                        existing_item.created_at
                    ),
                    "updated_at": (
                        existing_item.updated_at
                    ),
                },
            )

        memory_metadata = dict(metadata or {})

        if memory_type.strip():
            memory_metadata["type"] = (
                memory_type.strip()
            )

        memory_metadata["stage"] = str(
            self.current_stage
        )

        memory_id = str(uuid.uuid4())

        try:
            item = self._memory_manager.add_memory(
                memory_id = memory_id,
                content = content,
                metadata = memory_metadata,
            )
        except ValueError as error:
            return ToolResult(
                success = False,
                text = f"添加记忆失败: {error}",
                data = {
                    "error_type": type(error).__name__,
                },
            )

        return ToolResult(
            success = True,
            text = (
                "成功添加一条长期记忆。"
                f"Memory ID: {item.memory_id};"
                f"内容: {item.content}"
            ),
            data = {
                "memory_id": item.memory_id,
                "content": item.content,
                "metadata": dict(item.metadata),
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            },
        )

    def retrieve_memory(
            self,
            query: str,
            top_k: int = 3,
            metadata_filter: Optional[
                Dict[str, str]
            ] = None,
    ) -> ToolResult:
        """检索与查询语义相关的长期记忆。"""

        if not query.strip():
            return ToolResult(
                success = False,
                text = "检索记忆失败:query不能为空。"
            )

        try:
            normalized_top_k = int(top_k)
        except (TypeError, ValueError):
            return ToolResult(
                success=False,
                text="检索记忆失败:top_k必须是整数。",
            )

        if normalized_top_k <= 0:
            return ToolResult(
                success=False,
                text = "检索记忆失败:top_k必须大于0。"
            )

        items = self._memory_manager.retrieve(
            query = query,
            top_k = normalized_top_k,
            metadata_filter=metadata_filter,
        )

        if not items:
            return ToolResult(
                success=False,
                text = "没有找到相关记忆。",
                data = {
                    "query": query,
                    "memories": [],
                },
            )

        memory_data = [
            {
                "memory_id": item.memory_id,
                "content": item.content,
                "metadata": dict(
                    item.metadata
                ),
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in items
        ]

        memory_lines = [
            (
                f"- 内容: {item.content}\n"
                f"  Memory ID: "
                f"{item.memory_id}\n"
                f"  created_at: "
                f"{item.created_at}\n"
                f"  updated_at: "
                f"{item.updated_at}"
            )
            for item in items
        ]   

        text = (
            f"成功检索到{len(items)}条记忆：\n"
            +"\n".join(memory_lines)
        )

        return ToolResult(
            success = True,
            text = text,
            data = {
                "query": query,
                "memories": memory_data,
            },
        )

    def update_memory(
            self,
            memory_id: str,
            content: Optional[str] = None,
            metadata: Optional[
                Dict[str, str]
            ] = None,
    ) -> ToolResult:
        """更新已有记忆的正文或元数据。"""

        if not memory_id.strip():
            return ToolResult(
                success=False,
                text= "更新记忆失败:memory_id不能为空。",
            )

        if content is None and metadata is None:
            return ToolResult(
                success=False,
                text = (
                    "更新记忆失败:"
                    "content和metadata不能同时为空。"
                ),
                data = {
                    "memory_id": memory_id,
                },
            )

        try:
            updated = (
                self._memory_manager.update_memory(
                    memory_id = memory_id,
                    content = content,
                    metadata = metadata,
                )
            )
        except ValueError as error:
            return ToolResult(
                success=False,
                text = f"更新记忆失败: {error}",
                data = {
                    "memory_id": memory_id,
                    "error_type": type(error).__name__,
                },
            )

        if not updated:
            return ToolResult(
                success=False,
                text = (
                    "更新记忆失败："
                    f"找不到Memory ID {memory_id}。"
                ),
                data = {
                    "memory_id": memory_id,
                },
            )

        updated_item = (
            self._memory_manager.get_memory(
                memory_id
            )
        )

        return ToolResult(
            success=True,
            text=(
                "成功更新长期记忆。\n"
                f"Memory ID: {memory_id}\n"
                f"created_at: "
                f"{updated_item.created_at}\n"
                f"updated_at: "
                f"{updated_item.updated_at}"
            ),
            data = {
                "memory_id": memory_id,
                "content": (
                    updated_item.content
                    if updated_item is not None
                    else None
                ),
                "metadata": (
                    dict(updated_item.metadata)
                    if updated_item.metadata is not None
                    else {}
                ),
                "created_at": (
                    updated_item.created_at
                    if updated_item is not None
                    else None
                ),
                "updated_at": (
                    updated_item.updated_at
                    if updated_item is not None
                    else None
                ),
            },
        )

    def delete_memory(
            self,
            memory_id: str,
            confirmation: bool = False,
    ) -> ToolResult:
        
        if not memory_id.strip():
            return ToolResult(
                success=False,
                text="删除记忆失败:memory_id不能为空。",
            )

        if confirmation is not True:
            return ToolResult(
                success=False,
                text=(
                    "删除已取消："
                    "必须传入confirmation=True"
                    "才能永久删除记忆。"
                ),
                data={
                    "memory_id": memory_id,
                    "requires_confirmation": True,
                },
            )

        deleted = self._memory_manager.delete_memory(
            memory_id
        )

        if not deleted:
            return ToolResult(
                success=False,
                text=(
                    "删除记忆失败："
                    f"找不到Memory ID {memory_id}。"
                ),
                data={
                    "memory_id": memory_id,
                },
            )

        return ToolResult(
            success=True,
            text=(
                "成功删除长期记忆。"
                f"Memory ID：{memory_id}。"
            ),
            data={
                "memory_id": memory_id,
            },
        )

        