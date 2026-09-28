"""用 SQLite 记录飞书文件夹扫描结果，不下载、不解析、不写向量库。"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_feishu_connection import (
    FeishuProbeError,
    FeishuReadOnlyClient,
    _item_token,
    load_settings,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def metadata_fingerprint(folder_token: str, item: dict[str, Any]) -> str:
    """生成稳定的远端元数据指纹；正文哈希留给后续入库阶段。"""

    comparable = {
        "folder_token": folder_token,
        "token": _item_token(item),
        "name": str(item.get("name") or ""),
        "type": str(item.get("type") or "unknown"),
        "modified_time": str(
            item.get("modified_time") or item.get("modified_at") or ""
        ),
    }
    encoded = json.dumps(
        comparable, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class ScanSummary:
    discovered: int = 0
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    missing: int = 0
    jobs_created: int = 0


class FeishuSyncLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "FeishuSyncLedger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS sync_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,
                discovered_count INTEGER NOT NULL DEFAULT 0,
                new_count INTEGER NOT NULL DEFAULT 0,
                changed_count INTEGER NOT NULL DEFAULT 0,
                unchanged_count INTEGER NOT NULL DEFAULT 0,
                missing_count INTEGER NOT NULL DEFAULT 0,
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS documents (
                source_token TEXT PRIMARY KEY,
                folder_token TEXT NOT NULL,
                name TEXT NOT NULL,
                obj_type TEXT NOT NULL,
                remote_modified_time TEXT NOT NULL DEFAULT '',
                metadata_hash TEXT NOT NULL,
                content_hash TEXT,
                sync_status TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                last_changed_at TEXT NOT NULL,
                last_seen_run_id INTEGER NOT NULL,
                FOREIGN KEY(last_seen_run_id) REFERENCES sync_runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_documents_folder
            ON documents(folder_token);

            CREATE INDEX IF NOT EXISTS idx_documents_status
            ON documents(sync_status);

            CREATE TABLE IF NOT EXISTS sync_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_token TEXT NOT NULL,
                job_type TEXT NOT NULL CHECK(job_type IN ('upsert', 'mark_missing')),
                target_hash TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'running', 'completed', 'failed')),
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_error TEXT,
                UNIQUE(source_token, job_type, target_hash),
                FOREIGN KEY(source_token) REFERENCES documents(source_token)
            );

            CREATE INDEX IF NOT EXISTS idx_sync_jobs_status
            ON sync_jobs(status, created_at);
            """
        )
        self.connection.commit()

    def _enqueue_job(
        self,
        source_token: str,
        job_type: str,
        target_hash: str,
        created_at: str,
    ) -> bool:
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO sync_jobs(
                source_token, job_type, target_hash, status,
                attempts, created_at, updated_at
            ) VALUES (?, ?, ?, 'pending', 0, ?, ?)
            """,
            (source_token, job_type, target_hash, created_at, created_at),
        )
        return cursor.rowcount == 1

    def start_run(self, started_at: str) -> int:
        cursor = self.connection.execute(
            "INSERT INTO sync_runs(started_at, status) VALUES (?, 'running')",
            (started_at,),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def record_folder(
        self,
        run_id: int,
        folder_token: str,
        items: Iterable[dict[str, Any]],
        seen_at: str,
    ) -> ScanSummary:
        summary = ScanSummary()
        for item in items:
            source_token = _item_token(item)
            if not source_token:
                continue
            summary.discovered += 1
            fingerprint = metadata_fingerprint(folder_token, item)
            existing = self.connection.execute(
                "SELECT metadata_hash FROM documents WHERE source_token = ?",
                (source_token,),
            ).fetchone()

            name = str(item.get("name") or "未命名")
            obj_type = str(item.get("type") or "unknown")
            modified_time = str(
                item.get("modified_time") or item.get("modified_at") or ""
            )
            if existing is None:
                summary.new += 1
                self.connection.execute(
                    """
                    INSERT INTO documents(
                        source_token, folder_token, name, obj_type,
                        remote_modified_time, metadata_hash, sync_status,
                        first_seen_at, last_seen_at, last_changed_at,
                        last_seen_run_id
                    ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)
                    """,
                    (
                        source_token,
                        folder_token,
                        name,
                        obj_type,
                        modified_time,
                        fingerprint,
                        seen_at,
                        seen_at,
                        seen_at,
                        run_id,
                    ),
                )
            elif existing["metadata_hash"] != fingerprint:
                summary.changed += 1
                self.connection.execute(
                    """
                    UPDATE documents
                    SET folder_token = ?, name = ?, obj_type = ?,
                        remote_modified_time = ?, metadata_hash = ?,
                        sync_status = 'active', last_seen_at = ?,
                        last_changed_at = ?, last_seen_run_id = ?
                    WHERE source_token = ?
                    """,
                    (
                        folder_token,
                        name,
                        obj_type,
                        modified_time,
                        fingerprint,
                        seen_at,
                        seen_at,
                        run_id,
                        source_token,
                    ),
                )
            else:
                summary.unchanged += 1
                self.connection.execute(
                    """
                    UPDATE documents
                    SET sync_status = 'active', last_seen_at = ?, last_seen_run_id = ?
                    WHERE source_token = ?
                    """,
                    (seen_at, run_id, source_token),
                )
            if self._enqueue_job(
                source_token, "upsert", fingerprint, seen_at
            ):
                summary.jobs_created += 1
        self.connection.commit()
        return summary

    def mark_missing(
        self, run_id: int, folder_tokens: Iterable[str], changed_at: str
    ) -> tuple[int, int]:
        missing = 0
        jobs_created = 0
        for folder_token in folder_tokens:
            rows = self.connection.execute(
                """
                SELECT source_token, metadata_hash
                FROM documents
                WHERE folder_token = ?
                  AND last_seen_run_id <> ?
                  AND sync_status <> 'missing'
                """,
                (folder_token, run_id),
            ).fetchall()
            cursor = self.connection.execute(
                """
                UPDATE documents
                SET sync_status = 'missing', last_changed_at = ?
                WHERE folder_token = ?
                  AND last_seen_run_id <> ?
                  AND sync_status <> 'missing'
                """,
                (changed_at, folder_token, run_id),
            )
            missing += cursor.rowcount
            for row in rows:
                if self._enqueue_job(
                    row["source_token"],
                    "mark_missing",
                    row["metadata_hash"],
                    changed_at,
                ):
                    jobs_created += 1
        self.connection.commit()
        return missing, jobs_created

    def pending_job_counts(self) -> dict[str, int]:
        rows = self.connection.execute(
            """
            SELECT job_type, COUNT(*) AS count
            FROM sync_jobs
            WHERE status = 'pending'
            GROUP BY job_type
            ORDER BY job_type
            """
        ).fetchall()
        return {str(row["job_type"]): int(row["count"]) for row in rows}

    def finish_run(self, run_id: int, summary: ScanSummary, completed_at: str) -> None:
        self.connection.execute(
            """
            UPDATE sync_runs
            SET completed_at = ?, status = 'completed',
                discovered_count = ?, new_count = ?, changed_count = ?,
                unchanged_count = ?, missing_count = ?
            WHERE id = ?
            """,
            (
                completed_at,
                summary.discovered,
                summary.new,
                summary.changed,
                summary.unchanged,
                summary.missing,
                run_id,
            ),
        )
        self.connection.commit()

    def fail_run(self, run_id: int, error: str, completed_at: str) -> None:
        self.connection.execute(
            """
            UPDATE sync_runs
            SET completed_at = ?, status = 'failed', error = ?
            WHERE id = ?
            """,
            (completed_at, error[:1000], run_id),
        )
        self.connection.commit()


def resolve_ledger_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def scan_once() -> ScanSummary:
    settings = load_settings()
    raw_db_path = os.getenv(
        "FEISHU_SYNC_DB_PATH", "data/runtime/feishu_sync.db"
    )
    ledger_path = resolve_ledger_path(raw_db_path)
    if PROJECT_ROOT not in ledger_path.parents:
        raise FeishuProbeError("FEISHU_SYNC_DB_PATH 必须位于项目目录内")

    client = FeishuReadOnlyClient(settings.app_id, settings.app_secret)
    client.authenticate()
    started_at = utc_now()
    total = ScanSummary()

    with FeishuSyncLedger(ledger_path) as ledger:
        run_id = ledger.start_run(started_at)
        try:
            for folder_token in settings.folder_tokens:
                items = client.list_folder_items(folder_token)
                folder_summary = ledger.record_folder(
                    run_id, folder_token, items, utc_now()
                )
                total.discovered += folder_summary.discovered
                total.new += folder_summary.new
                total.changed += folder_summary.changed
                total.unchanged += folder_summary.unchanged
                total.jobs_created += folder_summary.jobs_created
            total.missing, missing_jobs = ledger.mark_missing(
                run_id, settings.folder_tokens, utc_now()
            )
            total.jobs_created += missing_jobs
            ledger.finish_run(run_id, total, utc_now())
            pending_jobs = ledger.pending_job_counts()
        except Exception as exc:
            ledger.fail_run(run_id, str(exc), utc_now())
            raise

    print(f"[成功] 同步台账扫描完成：{ledger_path}")
    print(
        "[结果] "
        f"发现 {total.discovered}，新增 {total.new}，变化 {total.changed}，"
        f"未变化 {total.unchanged}，缺失 {total.missing}，"
        f"新建任务 {total.jobs_created}"
    )
    print(
        "[队列] "
        f"待入库 {pending_jobs.get('upsert', 0)}，"
        f"待标记缺失 {pending_jobs.get('mark_missing', 0)}"
    )
    return total


def main() -> int:
    try:
        scan_once()
    except (FeishuProbeError, sqlite3.Error) as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
