import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


# 日志数据库放在 data/runtime 里。
# raw / processed 是资料，runtime 是项目运行时产生的数据。
PROJECT_DIR = Path(__file__).resolve().parent.parent
LOG_DB_PATH = PROJECT_DIR / "data" / "runtime" / "agent_logs.db"


def get_connection() -> sqlite3.Connection:
    """创建 SQLite 连接，并确保日志目录存在。"""

    LOG_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(LOG_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_log_db() -> None:
    """初始化日志表。

    这个函数可以重复调用。
    如果表已经存在，SQLite 不会重新创建。
    """

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                mode TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                trace_json TEXT NOT NULL,
                success INTEGER NOT NULL,
                error TEXT NOT NULL,
                latency_ms INTEGER NOT NULL
            )
            """
        )
        ensure_log_columns(connection)


def ensure_log_columns(connection: sqlite3.Connection) -> None:
    """老库补列：SQLite 没有 ADD COLUMN IF NOT EXISTS，重复加会报错，
    吞掉即可。intent / token 四列是 2026-09-21 加的（单次问答的
    意图与 token 计量，见 app/token_meter.py）。"""

    for statement in (
        "ALTER TABLE agent_logs ADD COLUMN intent TEXT",
        "ALTER TABLE agent_logs ADD COLUMN in_tokens INTEGER",
        "ALTER TABLE agent_logs ADD COLUMN out_tokens INTEGER",
        "ALTER TABLE agent_logs ADD COLUMN total_tokens INTEGER",
        "ALTER TABLE agent_logs ADD COLUMN llm_calls INTEGER",
        "ALTER TABLE agent_logs ADD COLUMN user_id INTEGER",
    ):
        try:
            connection.execute(statement)
        except sqlite3.OperationalError:
            pass  # 列已存在


def record_agent_log(
    *,
    mode: str,
    thread_id: str,
    question: str,
    answer: str,
    trace: Any,
    success: bool,
    error: str,
    latency_ms: int,
    intent: str | None = None,
    token_usage: dict[str, int] | None = None,
    user_id: int | None = None,
) -> int:
    """保存一次 Agent 请求日志，并返回日志 id。"""

    init_log_db()
    trace_json = json.dumps(trace, ensure_ascii=False, default=str)
    usage = token_usage or {}

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO agent_logs (
                created_at,
                mode,
                thread_id,
                question,
                answer,
                trace_json,
                success,
                error,
                latency_ms,
                intent,
                in_tokens,
                out_tokens,
                total_tokens,
                llm_calls,
                user_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                mode,
                thread_id,
                question,
                answer,
                trace_json,
                1 if success else 0,
                error,
                latency_ms,
                intent,
                usage.get("input_tokens"),
                usage.get("output_tokens"),
                usage.get("total_tokens"),
                usage.get("llm_calls"),
                user_id,
            ),
        )
        return int(cursor.lastrowid)


def list_agent_logs(limit: int = 20) -> list[dict[str, Any]]:
    """读取最近的 Agent 请求日志。"""

    init_log_db()
    safe_limit = max(1, min(int(limit), 100))

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                created_at,
                mode,
                thread_id,
                question,
                answer,
                trace_json,
                success,
                error,
                latency_ms,
                intent,
                in_tokens,
                out_tokens,
                total_tokens,
                llm_calls
            FROM agent_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

    logs = []
    for row in rows:
        item = dict(row)
        item["success"] = bool(item["success"])
        try:
            item["trace"] = json.loads(item.pop("trace_json"))
        except json.JSONDecodeError:
            item["trace"] = item.pop("trace_json")
        logs.append(item)

    return logs
