"""
负责数据的存储
"""

from datetime import (
    datetime,
    timezone,
)

from typing import (
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
)

import math
import threading
from dataclasses import dataclass, field

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算两个向量的余弦相似度"""

    if not a or not b:
        return 0.0

    if  len(a) != len(b):
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))

    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))

    if length_a == 0.0 or length_b == 0.0:
        return 0.0

    return dot_product / (length_a * length_b)

def utc_now_iso() -> str:
    """返回 ISO 8601 格式的UTC时间。"""

    return datetime.now(
        timezone.utc
    ).isoformat()

@dataclass
class MemoryItem:
    """一条长期记忆"""

    memory_id: str
    content: str
    metadata:Dict[str, str] = field(default_factory = dict)
    embedding: Optional[List[float]] = None
    
    created_at: str = field(
        default_factory = utc_now_iso
    )

    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.updated_at:
            self.updated_at = (
                self.created_at
            )

#! protocol表示：只要一个对象具有这些方法，就可以被当作MemoryStore使用
class MemoryStore(Protocol):
    """长期记忆存储需要遵守的接口"""

    def add(
            self,
            item: MemoryItem,
    ) -> None:
        ...

    def get(
            self,
            memory_id: str,
    ) -> Optional[MemoryItem]:
        ...

    def update(
            self,
            memory_id: str,
            new_content: Optional[str] = None,
            new_metadata: Optional[
                Dict[str, str]
            ] = None,
            new_embedding: Optional[
                List[float]
            ] = None,
    ) -> bool:
        ...

    def delete(
            self,
            memory_id: str,
    ) -> bool:
        ...

    def search(
            self,
            query_embedding: List[float],
            top_k: int = 5,
            metadata_filter: Optional[
                Dict[str, str]
            ] = None,
    ) -> List[
        Tuple[MemoryItem, float]
    ]:
        ...

    def get_all(
            self,
    ) -> List[MemoryItem]:
        ...

    def size(self) -> int:
        ...

    def clear(self) -> None:
        ...



class InMemoryVectorStore:
    """把记忆暂存在当前Python进程字典中"""

    def __init__(self) -> None:
        self._items: Dict[str, MemoryItem] = {}
        self._lock = threading.RLock()

    def add(self, item: MemoryItem) -> None:
        with self._lock:
            self._items[item.memory_id] = item

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        with self._lock:
            return self._items.get(memory_id)

    def update(
            self,
            memory_id: str,
            new_content: Optional[str] = None,
            new_metadata: Optional[Dict[str, str]] = None,
            new_embedding: Optional[List[float]] = None
    ) -> bool:
        with self._lock:
            item = self._items.get(memory_id)

            if item is None:
                return False

            if new_content is not None:
                item.content = new_content

            if new_metadata is not None:
                item.metadata.update(new_metadata) 

            if new_embedding is not None:
                item.embedding = new_embedding

            if (
                new_content is not None
                or new_metadata is not None
                or new_embedding is not None
            ):
                item.updated_at = (
                    utc_now_iso()
                )

            return True


    def delete(self, memory_id: str) -> bool:
        with self._lock:
            removed_item = self._items.pop(memory_id, None)
            return removed_item is not None
    
    def search(
            self,
            query_embedding: List[float],
            top_k: int = 5,
            metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[MemoryItem, float]]:
        with self._lock:
            scored_items: List[Tuple[MemoryItem, float]] = []

            for item in self._items.values():
                if item.embedding is None:
                    continue

                if metadata_filter is not None:
                    matches = all(
                        item.metadata.get(key) == value
                        for key, value in metadata_filter.items()
                    )

                    if not matches:
                        continue

                score = cosine_similarity(
                    query_embedding,
                    item.embedding,
                )

                if score > 0.0:
                    scored_items.append((item, score))

            scored_items.sort(
                key = lambda pair: pair[1],
                reverse = True
            )

            return scored_items[:max(1,top_k)]

    def get_all(self) -> List[MemoryItem]:
        with self._lock:
            return list(self._items.values())

    def size(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

