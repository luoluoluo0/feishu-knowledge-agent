"""飞书来源注册表的只读查询。

检索结果里只保存 16 位 ``item_id``，完整 file token 和原文链接只留在
SQLite。这里是检索层与同步台账之间唯一的桥，任何异常都只降级为“无链接”，
不能拖垮正常问答。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from app.config import PROJECT_DIR, get_settings


def _database_path() -> Path:
    path = Path(get_settings().feishu_sync_db_path)
    return path if path.is_absolute() else PROJECT_DIR / path


def lookup_source(item_id: str) -> dict[str, Any] | None:
    """按稳定 item_id 查飞书来源；同步库尚未创建时返回 ``None``。"""

    item_id = str(item_id or "").strip()
    path = _database_path()
    if not item_id or not path.exists():
        return None
    try:
        connection = sqlite3.connect(
            f"file:{path.as_posix()}?mode=ro", uri=True, timeout=0.2
        )
        connection.row_factory = sqlite3.Row
        try:
            row = connection.execute(
                """
                SELECT item_id, name, obj_type, source_url, last_synced_at
                FROM documents WHERE item_id=? LIMIT 1
                """,
                (item_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            connection.close()
    except (sqlite3.Error, OSError):
        return None
