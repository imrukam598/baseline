from pathlib import Path
from tempfile import TemporaryDirectory

from mini_agemem.memory import (
    MemoryItem,
)
from mini_agemem.sqlite_memory_store import (
    SQLiteMemoryStore,
)


def test_sqlite_memory_store() -> None:
    with TemporaryDirectory() as temp_dir:
        database_path = (
            Path(temp_dir)
            / "memories.db"
        )

        first_store = SQLiteMemoryStore(
            database_path
        )

        first_store.add(
            MemoryItem(
                memory_id="m1",
                content="用户喜欢 Python",
                metadata={
                    "type": "preference",
                },
                embedding=[
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            )
        )

        first_store.add(
            MemoryItem(
                memory_id="m2",
                content="用户计划去上海旅行",
                metadata={
                    "type": "plan",
                },
                embedding=[
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],
            )
        )

        assert first_store.size() == 2

        second_store = SQLiteMemoryStore(
            database_path
        )

        loaded_item = second_store.get(
            "m1"
        )

        assert loaded_item is not None
        assert loaded_item.content == (
            "用户喜欢 Python"
        )
        assert loaded_item.metadata == {
            "type": "preference",
        }
        assert loaded_item.embedding == [
            1.0,
            0.0,
            0.0,
            0.0,
        ]

        search_results = (
            second_store.search(
                query_embedding=[
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                top_k=1,
            )
        )

        assert len(search_results) == 1

        found_item, score = (
            search_results[0]
        )

        assert found_item.memory_id == "m1"
        assert score == 1.0

        updated = second_store.update(
            memory_id="m1",
            new_content=(
                "用户最喜欢 Python"
            ),
            new_metadata={
                "status": "updated",
            },
            new_embedding=[
                1.0,
                0.0,
                0.0,
                0.0,
            ],
        )

        assert updated is True

        updated_item = second_store.get(
            "m1"
        )

        assert updated_item is not None
        assert updated_item.content == (
            "用户最喜欢 Python"
        )
        assert updated_item.metadata == {
            "type": "preference",
            "status": "updated",
        }

        deleted = second_store.delete(
            "m2"
        )

        assert deleted is True
        assert second_store.size() == 1

        print(
            "SQLite 长期记忆持久化测试通过"
        )


if __name__ == "__main__":
    test_sqlite_memory_store()