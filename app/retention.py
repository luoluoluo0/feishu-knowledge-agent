import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

from app import logging_store
from app.checkpoint_store import table_exists
from app.config import Settings


logger = logging.getLogger(__name__)


# 数据保留策略：三个 SQLite 库（请求日志 / 会话轮次 / LangGraph checkpoint）
# 的统一清理入口。之前这些库只增不减——每次评测 95 个线程、每线程带完整
# 检索历史的 checkpoint，长期运行吃掉几百 MB 只是时间问题。
#
# 清理顺序有一个关键约束：checkpoint 的「不活跃」判断依赖 conversation_turns
# 的 created_at（LangGraph 自己的 checkpoint_id 时间格式脆弱，不解析），
# 所以必须**先收集旧线程、再删 checkpoints、最后删 turns**——顺序反了就
# 永远找不到要清的线程。没有对应 turns 的线程（异常中断产生的孤儿）保守保留。


def _cutoff(settings: Settings) -> str:
    """ISO 格式的保留线。created_at 存的是 ISO 文本，字符串比较即时间比较。"""

    return (
        datetime.now() - timedelta(days=settings.data_retention_days)
    ).isoformat(timespec="seconds")


def _connect(db_path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _cleanup_logs(log_db_path, archive_dir: str, cutoff: str) -> int:
    """归档并删除过期请求日志，返回删除条数。"""

    if not Path(log_db_path).exists():
        return 0

    with _connect(log_db_path) as connection:
        if not table_exists(connection, "agent_logs"):
            return 0

        rows = connection.execute(
            "SELECT * FROM agent_logs WHERE created_at < ?", (cutoff,)
        ).fetchall()
        if not rows:
            return 0

        # 先归档再删除。按月分文件、追加写入，人工回看和导入分析都方便。
        archive = Path(archive_dir) / f"agent_logs_{datetime.now():%Y%m}.jsonl"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with archive.open("a", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")

        cursor = connection.execute(
            "DELETE FROM agent_logs WHERE created_at < ?", (cutoff,)
        )
        return cursor.rowcount


def _collect_stale_thread_ids(turns_db_path, cutoff: str) -> set[str]:
    """收集「最新一轮早于 cutoff」的线程。turns 库不存在时返回空集。"""

    if not Path(turns_db_path).exists():
        return set()

    with _connect(turns_db_path) as connection:
        if not table_exists(connection, "conversation_turns"):
            return set()
        rows = connection.execute(
            """
            SELECT thread_id FROM conversation_turns
            GROUP BY thread_id HAVING MAX(created_at) < ?
            """,
            (cutoff,),
        ).fetchall()
    return {row["thread_id"] for row in rows}


def _cleanup_turns(turns_db_path, cutoff: str) -> int:
    if not Path(turns_db_path).exists():
        return 0

    with _connect(turns_db_path) as connection:
        if not table_exists(connection, "conversation_turns"):
            return 0
        cursor = connection.execute(
            "DELETE FROM conversation_turns WHERE created_at < ?", (cutoff,)
        )
        return cursor.rowcount


def _cleanup_checkpoints(ckpt_db_path, thread_ids: set[str]) -> int:
    """删除指定线程的 LangGraph checkpoint。库/表不存在或列表为空则跳过。"""

    if not thread_ids or not Path(ckpt_db_path).exists():
        return 0

    deleted = 0
    with _connect(ckpt_db_path) as connection:
        placeholders = ",".join("?" for _ in thread_ids)
        params = tuple(thread_ids)
        for table in ("writes", "checkpoints"):
            if table_exists(connection, table):
                cursor = connection.execute(
                    f"DELETE FROM {table} WHERE thread_id IN ({placeholders})",
                    params,
                )
                deleted += cursor.rowcount
    return deleted


def run_retention_cleanup(settings: Settings) -> dict[str, int]:
    """跑一轮完整清理，返回各项计数。幂等：重复跑第二轮全为 0。"""

    cutoff = _cutoff(settings)
    stale_threads = _collect_stale_thread_ids(settings.conversation_db_path, cutoff)

    checkpoint_rows = _cleanup_checkpoints(
        settings.checkpoint_db_path, stale_threads
    )
    turns = _cleanup_turns(settings.conversation_db_path, cutoff)
    logs = _cleanup_logs(
        logging_store.LOG_DB_PATH, settings.retention_archive_dir, cutoff
    )

    counts = {
        "logs_deleted": logs,
        "turns_deleted": turns,
        "checkpoint_rows_deleted": checkpoint_rows,
        "checkpoint_threads": len(stale_threads),
    }
    if any(counts.values()):
        logger.info("数据保留清理完成：%s", counts)
    return counts


_retention_thread: threading.Thread | None = None


def start_retention_daemon(settings: Settings, interval_seconds: int = 86400) -> None:
    """启动后台守护线程，每 interval_seconds 跑一次清理。

    启动时已经跑过一次，所以这里先睡再跑。进程退出时随 daemon 线程终止。
    """

    global _retention_thread

    def _loop():
        import time

        while True:
            time.sleep(interval_seconds)
            try:
                run_retention_cleanup(settings)
            except Exception:
                logger.warning("周期性数据清理失败：", exc_info=True)

    if _retention_thread and _retention_thread.is_alive():
        return
    _retention_thread = threading.Thread(target=_loop, daemon=True, name="retention")
    _retention_thread.start()
