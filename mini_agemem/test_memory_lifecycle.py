import json
import sqlite3
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from mini_agemem.sqlite_memory_store import (
    SQLiteMemoryStore,
)


def test_memory_lifecycle() -> None:
    with TemporaryDirectory() as temp_dir:
        database_path = (
            Path(temp_dir)
            / "legacy_memories.db"
        )

        connection = sqlite3.connect(
            database_path
        )

        try:
            connection.execute(
                """
                CREATE TABLE memories (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                INSERT INTO memories (
                    memory_id,
                    content,
                    metadata_json,
                    embedding_json
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    "legacy-1",
                    "一条旧版本记忆",
                    json.dumps({
                        "type": "legacy",
                    }),
                    json.dumps([
                        1.0,
                        0.0,
                    ]),
                ),
            )

            connection.commit()
        finally:
            connection.close()

        store = SQLiteMemoryStore(
            database_path
        )

        migrated_item = store.get(
            "legacy-1"
        )

        assert migrated_item is not None
        assert migrated_item.created_at
        assert migrated_item.updated_at

        original_created_at = (
            migrated_item.created_at
        )
        original_updated_at = (
            migrated_item.updated_at
        )

        time.sleep(0.01)

        updated = store.update(
            memory_id="legacy-1",
            new_content=(
                "更新后的旧版本记忆"
            ),
            new_embedding=[
                0.0,
                1.0,
            ],
        )

        assert updated is True

        updated_item = store.get(
            "legacy-1"
        )

        assert updated_item is not None

        assert (
            updated_item.created_at
            == original_created_at
        )

        assert (
            updated_item.updated_at
            > original_updated_at
        )

        print(
            "长期记忆生命周期测试通过"
        )


if __name__ == "__main__":
    test_memory_lifecycle()