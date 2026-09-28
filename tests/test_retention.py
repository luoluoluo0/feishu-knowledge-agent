"""数据保留策略测试。纯本地临时 SQLite，零外部依赖。"""

import json
import sqlite3

import pytest

from app import logging_store
from app.retention import (
    _cleanup_checkpoints,
    _cleanup_logs,
    _cleanup_turns,
    _collect_stale_thread_ids,
    run_retention_cleanup,
)


CUTOFF = "2026-09-01T00:00:00"
OLD = "2026-06-01T00:00:00"  # 早于 cutoff
NEW = "2026-09-10T00:00:00"  # 晚于 cutoff


def _exec(db_path, statements):
    with sqlite3.connect(db_path) as connection:
        for statement in statements:
            connection.execute(statement)


def seed_logs(db_path, old_count=2, new_count=1):
    _exec(
        db_path,
        [
            """CREATE TABLE agent_logs (
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
            )""",
        ]
        + [
            f"INSERT INTO agent_logs (created_at, mode, thread_id, question, answer, trace_json, success, error, latency_ms) "
            f"VALUES ('{OLD if i < old_count else NEW}', 'tool_agent', 't', 'q', 'a', '{{}}', 1, '', 1)"
            for i in range(old_count + new_count)
        ],
    )


def seed_turns(db_path, rows):
    """rows: [(thread_id, created_at)]"""
    _exec(db_path, ["CREATE TABLE conversation_turns (id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT, created_at TEXT)"])
    with sqlite3.connect(db_path) as connection:
        for thread_id, created_at in rows:
            connection.execute(
                "INSERT INTO conversation_turns (thread_id, created_at) VALUES (?, ?)",
                (thread_id, created_at),
            )


def seed_checkpoints(db_path, thread_ids):
    _exec(
        db_path,
        [
            "CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT)",
            "CREATE TABLE writes (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT)",
        ],
    )
    with sqlite3.connect(db_path) as connection:
        for thread_id in thread_ids:
            for table in ("checkpoints", "writes"):
                connection.execute(
                    f"INSERT INTO {table} (thread_id, checkpoint_ns, checkpoint_id) VALUES (?, '', 'c1')",
                    (thread_id,),
                )


def count_rows(db_path, table):
    with sqlite3.connect(db_path) as connection:
        return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


class TestLogs:
    def test_old_archived_and_deleted_new_kept(self, tmp_path):
        db = tmp_path / "logs.db"
        archive_dir = tmp_path / "archive"
        seed_logs(db, old_count=2, new_count=1)

        deleted = _cleanup_logs(db, str(archive_dir), CUTOFF)

        assert deleted == 2
        assert count_rows(db, "agent_logs") == 1  # 新的那条还在
        # 归档文件存在且内容是两条旧记录
        archives = list(archive_dir.glob("agent_logs_*.jsonl"))
        assert len(archives) == 1
        lines = [json.loads(line) for line in archives[0].read_text(encoding="utf-8").splitlines()]
        assert len(lines) == 2
        assert all(row["created_at"] == OLD for row in lines)

    def test_nothing_to_delete(self, tmp_path):
        db = tmp_path / "logs.db"
        seed_logs(db, old_count=0, new_count=1)
        assert _cleanup_logs(db, str(tmp_path / "archive"), CUTOFF) == 0
        # 没有过期数据时不产生空归档文件
        assert not (tmp_path / "archive").exists() or not list((tmp_path / "archive").glob("*.jsonl"))


class TestTurns:
    def test_old_deleted_new_kept(self, tmp_path):
        db = tmp_path / "turns.db"
        seed_turns(db, [("old-thread", OLD), ("old-thread", OLD), ("new-thread", NEW)])

        deleted = _cleanup_turns(db, CUTOFF)

        assert deleted == 2
        with sqlite3.connect(db) as connection:
            remaining = {row[0] for row in connection.execute("SELECT thread_id FROM conversation_turns")}
        assert remaining == {"new-thread"}


class TestCheckpoints:
    def test_only_stale_threads_cleaned(self, tmp_path):
        turns_db = tmp_path / "turns.db"
        ckpt_db = tmp_path / "ckpt.db"
        seed_turns(turns_db, [("old-thread", OLD), ("new-thread", NEW)])
        seed_checkpoints(ckpt_db, ["old-thread", "new-thread", "orphan-thread"])

        stale = _collect_stale_thread_ids(turns_db, CUTOFF)
        assert stale == {"old-thread"}

        deleted = _cleanup_checkpoints(ckpt_db, stale)
        assert deleted == 2  # checkpoints 1 条 + writes 1 条
        # old-thread 清掉，new-thread 和孤儿保守保留
        assert count_rows(ckpt_db, "checkpoints") == 2
        assert count_rows(ckpt_db, "writes") == 2

    def test_order_safe(self, tmp_path):
        """先删 turns 再收集会导致旧线程找不到——公共入口必须保证顺序。"""
        turns_db = tmp_path / "turns.db"
        ckpt_db = tmp_path / "ckpt.db"
        seed_turns(turns_db, [("old-thread", OLD)])
        seed_checkpoints(ckpt_db, ["old-thread"])

        settings = type("S", (), {
            "data_retention_days": 90,
            "retention_archive_dir": str(tmp_path / "archive"),
            "conversation_db_path": str(turns_db),
            "checkpoint_db_path": str(ckpt_db),
        })()
        counts = run_retention_cleanup(settings)

        assert counts["turns_deleted"] == 1
        assert counts["checkpoint_rows_deleted"] == 2
        assert count_rows(ckpt_db, "checkpoints") == 0

    def test_idempotent(self, tmp_path):
        turns_db = tmp_path / "turns.db"
        ckpt_db = tmp_path / "ckpt.db"
        seed_turns(turns_db, [("old-thread", OLD)])
        seed_checkpoints(ckpt_db, ["old-thread"])
        settings = type("S", (), {
            "data_retention_days": 90,
            "retention_archive_dir": str(tmp_path / "archive"),
            "conversation_db_path": str(turns_db),
            "checkpoint_db_path": str(ckpt_db),
        })()

        run_retention_cleanup(settings)
        counts2 = run_retention_cleanup(settings)

        assert counts2["turns_deleted"] == 0
        assert counts2["checkpoint_rows_deleted"] == 0

    def test_missing_dbs_are_noop(self, tmp_path):
        settings = type("S", (), {
            "data_retention_days": 90,
            "retention_archive_dir": str(tmp_path / "archive"),
            "conversation_db_path": str(tmp_path / "missing_turns.db"),
            "checkpoint_db_path": str(tmp_path / "missing_ckpt.db"),
        })()
        monkey_log = tmp_path / "missing_logs.db"
        original = logging_store.LOG_DB_PATH
        logging_store.LOG_DB_PATH = monkey_log
        try:
            counts = run_retention_cleanup(settings)
        finally:
            logging_store.LOG_DB_PATH = original
        assert counts == {
            "logs_deleted": 0,
            "turns_deleted": 0,
            "checkpoint_rows_deleted": 0,
            "checkpoint_threads": 0,
        }
