import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from app.config import Settings, get_settings
from app.coreference import Turn


# 会话历史存储：只保存查询改写需要的最小上下文。
#
# 与 LangGraph checkpointer 分开存放。checkpointer 保存 Agent 循环用的
# 完整消息状态，这里是给改写模块看的历史，两者用途不同，共用一个
# SQLite 文件会在 Windows 上增加文件锁冲突的概率
# （见 app/agent.py 中 close_checkpointers 的注释）。


logger = logging.getLogger(__name__)

MAX_ANSWER_CHARS = 500

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    question TEXT NOT NULL,
    rewritten_query TEXT NOT NULL DEFAULT '',
    item_id TEXT NOT NULL DEFAULT '',
    answer TEXT NOT NULL DEFAULT '',
    intent TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
)
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_conversation_turns_thread
    ON conversation_turns (thread_id, id DESC)
"""

# 老库没有 intent 列时的迁移。列已存在时 sqlite 会抛
# "duplicate column name"，按迁移完成处理。
MIGRATE_INTENT_SQL = (
    "ALTER TABLE conversation_turns ADD COLUMN intent TEXT NOT NULL DEFAULT ''"
)


def get_db_path(settings: Settings | None = None) -> Path:
    """返回会话历史数据库路径。"""

    settings = settings or get_settings()
    return Path(settings.conversation_db_path)


@contextmanager
def open_db(settings: Settings | None = None) -> Iterator[sqlite3.Connection]:
    """打开数据库并确保表结构存在。

    正常退出时提交，无论成功失败都关闭连接。
    """

    db_path = get_db_path(settings)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    try:
        connection.execute(CREATE_TABLE_SQL)
        try:
            connection.execute(MIGRATE_INTENT_SQL)
        except sqlite3.OperationalError:
            pass  # 列已存在
        connection.execute(CREATE_INDEX_SQL)
        yield connection
        connection.commit()
    finally:
        connection.close()


def load_history(
    thread_id: str,
    limit: int = 5,
    settings: Settings | None = None,
) -> list[Turn]:
    """读取最近 limit 轮历史，按时间正序返回。"""

    thread_id = thread_id.strip()
    if not thread_id:
        return []

    try:
        with open_db(settings) as connection:
            rows = connection.execute(
                """
                SELECT question, rewritten_query, item_id, answer, intent
                FROM conversation_turns
                WHERE thread_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (thread_id, limit),
            ).fetchall()
    except Exception as exc:
        logger.warning("会话历史读取失败，按无历史处理：%s", exc)
        return []

    return [
        Turn(
            question=row[0],
            rewritten_query=row[1],
            item_id=row[2],
            answer=row[3],
            intent=row[4] or "",
        )
        for row in reversed(rows)
    ]


def append_turn(
    thread_id: str,
    turn: Turn,
    settings: Settings | None = None,
) -> None:
    """写入一轮对话。写入失败只记录警告，不中断主流程。"""

    thread_id = thread_id.strip()
    if not thread_id:
        logger.warning("thread_id 为空，跳过会话历史写入。")
        return

    try:
        with open_db(settings) as connection:
            connection.execute(
                """
                INSERT INTO conversation_turns
                    (thread_id, question, rewritten_query, item_id, answer, intent, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    turn.question,
                    turn.rewritten_query,
                    turn.item_id,
                    turn.answer[:MAX_ANSWER_CHARS],
                    turn.intent or "",
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    except Exception as exc:
        logger.warning("会话历史写入失败，本轮不记录：%s", exc)


def clear_history(thread_id: str, settings: Settings | None = None) -> int:
    """清空某个 thread_id 的全部历史，返回删除行数。"""

    thread_id = thread_id.strip()
    if not thread_id:
        return 0

    try:
        with open_db(settings) as connection:
            cursor = connection.execute(
                "DELETE FROM conversation_turns WHERE thread_id = ?",
                (thread_id,),
            )
            return cursor.rowcount
    except Exception as exc:
        logger.warning("会话历史清理失败：%s", exc)
        return 0
