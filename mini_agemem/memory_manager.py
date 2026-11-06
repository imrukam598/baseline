"""
负责向量化和业务协调
"""
from typing import Dict, List, Optional
from mini_agemem.memory import (
    MemoryItem,
    MemoryStore,
    InMemoryVectorStore,
)

from mini_agemem.embedding import EmbeddingClient

class MemoryManager:
    """协调文本向量化和记忆存储"""

    def __init__(
            self,
            embedding_client: EmbeddingClient,
            store: Optional[MemoryStore] = None,
    ) -> None:

        self._embedding_client = embedding_client

        if store is None:
            self._store = InMemoryVectorStore()
        else:
            self._store = store

    def find_exact_memory(
            self,
            content: str,
    ) -> Optional[MemoryItem]:
        """查找完全重复的记忆"""
        normalized_content = " ".join(
            content.casefold().split()
        )

        for item in self._store.get_all():
            normalized_item_content = (
                " ".join(
                    item.content
                    .casefold()
                    .split()
                )
            )

            if (
                normalized_item_content
                == normalized_content
            ):
                return item

        return None

    def add_memory(
            self,
            memory_id,
            content: str,
            metadata: Optional[Dict[str, str]] = None
    ) -> MemoryItem:
        
        if not memory_id:
            raise ValueError("memory_id不能为空")

        if not content.strip():
            raise ValueError("content不能为空")

        embedding = self._embedding_client.embed(content)

        item = MemoryItem(
            memory_id = memory_id,
            content = content,
            metadata = metadata or {},
            embedding = embedding,
        )

        self._store.add(item)

        return item

    def retrieve(
            self,
            query: str,
            top_k: int = 5,
            metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[MemoryItem]:
        if not query.strip():
            return []

        query_embedding = self._embedding_client.embed(query)

        scored_items = self._store.search(
            query_embedding = query_embedding,
            top_k = top_k,
            metadata_filter = metadata_filter,
        )

        return [
            item
            for item, score in scored_items
        ]

    def update_memory(
            self,
            memory_id: str,
            content: Optional[str] = None,
            metadata: Optional[Dict[str, str]] = None,
    ) -> bool:
        if content is not None:
            if not content.strip():
                raise ValueError("content不能为空")

            new_embedding = self._embedding_client.embed(content)
        else:
            new_embedding = None

        return self._store.update(
            memory_id = memory_id,
            new_content = content,
            new_metadata = metadata,
            new_embedding = new_embedding, 
        )

    def delete_memory(self, memory_id: str) -> bool:
        return self._store.delete(memory_id)

    #! Optional[MemoryItem] 可以是MemoryItem也可以是None
    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        return self._store.get(memory_id)

    def get_all_memory(self) -> List[MemoryItem]:
        return self._store.get_all()

    def size(self) -> int:
        return self._store.size()

    def clear(self) -> None:
        self._store.clear()

    
