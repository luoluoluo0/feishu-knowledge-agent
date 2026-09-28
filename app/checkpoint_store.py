import sqlite3
from pathlib import Path

from app.config import Settings, get_settings


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    """判断 checkpoint 数据库里是否存在某张表。"""

    row = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def clear_thread_checkpoints(
    thread_id: str,
    settings: Settings | None = None,
) -> dict[str, int | str | bool]:
    """删除某个 thread_id 对应的 LangGraph checkpoint 记忆。"""

    thread_id = thread_id.strip()
    if not thread_id:
        raise ValueError("thread_id 不能为空。")

    settings = settings or get_settings()
    db_path = Path(settings.checkpoint_db_path)

    if not db_path.exists():
        return {
            "thread_id": thread_id,
            "db_path": str(db_path),
            "db_exists": False,
            "deleted_checkpoints": 0,
            "deleted_writes": 0,
        }

    with sqlite3.connect(db_path) as connection:
        deleted_writes = 0
        deleted_checkpoints = 0

        if table_exists(connection, "writes"):
            cursor = connection.execute(
                "DELETE FROM writes WHERE thread_id = ?",
                (thread_id,),
            )
            deleted_writes = cursor.rowcount

        if table_exists(connection, "checkpoints"):
            cursor = connection.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            )
            deleted_checkpoints = cursor.rowcount

    return {
        "thread_id": thread_id,
        "db_path": str(db_path),
        "db_exists": True,
        "deleted_checkpoints": deleted_checkpoints,
        "deleted_writes": deleted_writes,
    }
