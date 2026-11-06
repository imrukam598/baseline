import json
import sqlite3
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

from mini_agemem.memory import (
    MemoryItem,
    cosine_similarity,
    utc_now_iso,
)


class SQLiteMemoryStore:
    """把长期记忆保存在 SQLite 数据库中。"""

    def __init__(
            self,
            database_path: str | Path,
    ) -> None:
        self._database_path = Path(
            database_path
        )

        self._database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._create_table()

    def _connect(
            self,
    ) -> sqlite3.Connection:
        return sqlite3.connect(
            self._database_path
        )

    def _create_table(self) -> None:
        connection = self._connect()

        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            cursor = connection.execute(
                """
                PRAGMA table_info(memories)
                """
            )

            column_names = {
                row[1]
                for row in cursor.fetchall()
            }

            if "created_at" not in column_names:
                connection.execute(
                    """
                    ALTER TABLE memories
                    ADD COLUMN created_at
                    TEXT NOT NULL DEFAULT ''
                    """
                )

            if "updated_at" not in column_names:
                connection.execute(
                    """
                    ALTER TABLE memories
                    ADD COLUMN updated_at
                    TEXT NOT NULL DEFAULT ''
                    """
                )

            migration_time = utc_now_iso()

            connection.execute(
                """
                UPDATE memories
                SET created_at = ?
                WHERE created_at = ''
                """,
                (migration_time,),
            )

            connection.execute(
                """
                UPDATE memories
                SET updated_at = created_at
                WHERE updated_at = ''
                """
            )

            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _row_to_item(
            row: tuple,
    ) -> MemoryItem:
        return MemoryItem(
            memory_id=row[0],
            content=row[1],
            metadata=json.loads(row[2]),
            embedding=json.loads(row[3]),
            created_at=row[4],
            updated_at=row[5],
        )
    
    def add(
            self,
            item: MemoryItem,
    ) -> None:
        connection = self._connect()

        try:
            connection.execute(
                """
                INSERT OR REPLACE INTO memories (
                    memory_id,
                    content,
                    metadata_json,
                    embedding_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    item.memory_id,
                    item.content,
                    json.dumps(
                        item.metadata,
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        item.embedding,
                    ),
                    item.created_at,
                    item.updated_at,
                ),
            )

            connection.commit()
        finally:
            connection.close()

    def get(
            self,
            memory_id: str,
    ) -> Optional[MemoryItem]:
        connection = self._connect()

        try:
            cursor = connection.execute(
                """
                SELECT
                    memory_id,
                    content,
                    metadata_json,
                    embedding_json,
                    created_at,
                    updated_at
                FROM memories
                WHERE memory_id = ?
                """,
                (memory_id,),
            )

            row = cursor.fetchone()
        finally:
            connection.close()

        if row is None:
            return None

        return self._row_to_item(row)

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
        item = self.get(memory_id)

        if item is None:
            return False

        if new_content is not None:
            item.content = new_content

        if new_metadata is not None:
            item.metadata.update(
                new_metadata
            )

        if new_embedding is not None:
            item.embedding = new_embedding

        if (
            new_content is not None
            or new_metadata is not None
            or new_embedding is not None
        ):
            item.updated_at = utc_now_iso()

        self.add(item)

        return True

    def delete(
            self,
            memory_id: str,
    ) -> bool:
        connection = self._connect()

        try:
            cursor = connection.execute(
                """
                DELETE FROM memories
                WHERE memory_id = ?
                """,
                (memory_id,),
            )

            connection.commit()

            deleted = cursor.rowcount > 0
        finally:
            connection.close()

        return deleted

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
        scored_items: List[
            Tuple[MemoryItem, float]
        ] = []

        for item in self.get_all():
            if item.embedding is None:
                continue

            if metadata_filter is not None:
                matches = all(
                    item.metadata.get(key)
                    == value
                    for key, value
                    in metadata_filter.items()
                )

                if not matches:
                    continue

            score = cosine_similarity(
                query_embedding,
                item.embedding,
            )

            if score > 0.0:
                scored_items.append(
                    (item, score)
                )

        scored_items.sort(
            key=lambda pair: pair[1],
            reverse=True,
        )

        return scored_items[
            :max(1, top_k)
        ]

    def get_all(
            self,
    ) -> List[MemoryItem]:
        connection = self._connect()

        try:
            cursor = connection.execute(
                """         
                SELECT
                    memory_id,
                    content,
                    metadata_json,
                    embedding_json,
                    created_at,
                    updated_at
                FROM memories
                ORDER BY memory_id
                """
            )

            rows = cursor.fetchall()
        finally:
            connection.close()

        return [
            self._row_to_item(row)
            for row in rows
        ]

    def size(self) -> int:
        connection = self._connect()

        try:
            cursor = connection.execute(
                """
                SELECT COUNT(*)
                FROM memories
                """
            )

            row = cursor.fetchone()
        finally:
            connection.close()

        assert row is not None

        return int(row[0])

    def clear(self) -> None:
        connection = self._connect()

        try:
            connection.execute(
                """
                DELETE FROM memories
                """
            )

            connection.commit()
        finally:
            connection.close()