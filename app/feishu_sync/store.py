from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from .models import SyncJob


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utc_after(seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat(
        timespec="seconds"
    )


def stable_item_id(tenant_key: str, source_token: str) -> str:
    return hashlib.sha256(f"{tenant_key}:{source_token}".encode("utf-8")).hexdigest()[:16]


def metadata_hash(root_token: str, parent_token: str, item: dict[str, Any]) -> str:
    payload = {
        "root_token": root_token,
        "parent_token": parent_token,
        "source_token": str(item.get("token") or item.get("file_token") or ""),
        "name": str(item.get("name") or ""),
        "type": str(item.get("type") or "unknown"),
        "modified_time": str(item.get("modified_time") or item.get("modified_at") or ""),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class SyncStore:
    RETRY_DELAYS = (60, 300, 1800, 7200)

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SyncStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @contextmanager
    def immediate_transaction(self) -> Iterator[sqlite3.Connection]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except Exception:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def _migrate(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sources (
                tenant_key TEXT NOT NULL,
                root_token TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'folder',
                enabled INTEGER NOT NULL DEFAULT 1,
                last_status TEXT NOT NULL DEFAULT 'new',
                last_error TEXT,
                last_scan_at TEXT,
                PRIMARY KEY(tenant_key, root_token)
            );

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
                folder_token TEXT NOT NULL DEFAULT '',
                name TEXT NOT NULL DEFAULT '',
                obj_type TEXT NOT NULL DEFAULT 'unknown',
                remote_modified_time TEXT NOT NULL DEFAULT '',
                metadata_hash TEXT NOT NULL DEFAULT '',
                content_hash TEXT,
                sync_status TEXT NOT NULL DEFAULT 'active',
                first_seen_at TEXT NOT NULL DEFAULT '',
                last_seen_at TEXT NOT NULL DEFAULT '',
                last_changed_at TEXT NOT NULL DEFAULT '',
                last_seen_run_id INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS sync_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_token TEXT NOT NULL,
                job_type TEXT NOT NULL,
                target_hash TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_error TEXT,
                UNIQUE(source_token, job_type, target_hash)
            );

            CREATE TABLE IF NOT EXISTS sync_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                source_token TEXT NOT NULL DEFAULT '',
                received_at TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS corpus_state (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                revision INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            """
        )
        document_columns = {
            "tenant_key": "TEXT NOT NULL DEFAULT ''",
            "root_token": "TEXT NOT NULL DEFAULT ''",
            "parent_folder_token": "TEXT NOT NULL DEFAULT ''",
            "item_id": "TEXT NOT NULL DEFAULT ''",
            "source_url": "TEXT NOT NULL DEFAULT ''",
            "remote_version": "TEXT NOT NULL DEFAULT ''",
            "missing_count": "INTEGER NOT NULL DEFAULT 0",
            "soft_deleted_at": "TEXT",
            "last_synced_at": "TEXT",
            "last_error": "TEXT",
        }
        job_columns = {
            "lease_owner": "TEXT NOT NULL DEFAULT ''",
            "lease_until": "TEXT",
            "next_retry_at": "TEXT",
            "completed_at": "TEXT",
        }
        for name, definition in document_columns.items():
            self._ensure_column("documents", name, definition)
        for name, definition in job_columns.items():
            self._ensure_column("sync_jobs", name, definition)
        self.connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_root_seen
            ON documents(tenant_key, root_token, last_seen_run_id);
            CREATE INDEX IF NOT EXISTS idx_documents_item_id
            ON documents(item_id);
            CREATE INDEX IF NOT EXISTS idx_documents_status
            ON documents(sync_status);
            CREATE INDEX IF NOT EXISTS idx_sync_jobs_claim
            ON sync_jobs(status, next_retry_at, created_at);
            INSERT OR IGNORE INTO corpus_state(singleton, revision, updated_at)
            VALUES (1, 0, CURRENT_TIMESTAMP);
            INSERT OR IGNORE INTO schema_migrations(version, applied_at)
            VALUES (1, CURRENT_TIMESTAMP);
            """
        )
        self.connection.commit()

    def _ensure_column(self, table: str, name: str, definition: str) -> None:
        existing = {
            str(row["name"])
            for row in self.connection.execute(f"PRAGMA table_info({table})")
        }
        if name not in existing:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    def start_run(self) -> int:
        cursor = self.connection.execute(
            "INSERT INTO sync_runs(started_at, status) VALUES (?, 'running')",
            (utc_now(),),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def finish_run(self, run_id: int, counts: dict[str, int], error: str = "") -> None:
        status = "failed" if error else "completed"
        self.connection.execute(
            """
            UPDATE sync_runs SET completed_at=?, status=?, discovered_count=?,
                new_count=?, changed_count=?, unchanged_count=?, missing_count=?, error=?
            WHERE id=?
            """,
            (
                utc_now(),
                status,
                counts.get("discovered", 0),
                counts.get("new", 0),
                counts.get("changed", 0),
                counts.get("unchanged", 0),
                counts.get("missing", 0),
                error[:1000] or None,
                run_id,
            ),
        )
        self.connection.commit()

    def ensure_source(self, tenant_key: str, root_token: str) -> None:
        self.connection.execute(
            """
            INSERT INTO sources(tenant_key, root_token) VALUES (?, ?)
            ON CONFLICT(tenant_key, root_token) DO UPDATE SET enabled=1
            """,
            (tenant_key, root_token),
        )
        self.connection.commit()

    def set_source_status(
        self, tenant_key: str, root_token: str, status: str, error: str = ""
    ) -> None:
        self.connection.execute(
            """
            UPDATE sources SET last_status=?, last_error=?, last_scan_at=?
            WHERE tenant_key=? AND root_token=?
            """,
            (status, error[:1000] or None, utc_now(), tenant_key, root_token),
        )
        self.connection.commit()

    def record_document(
        self,
        *,
        run_id: int,
        tenant_key: str,
        root_token: str,
        parent_folder_token: str,
        source_token: str,
        name: str,
        obj_type: str,
        source_url: str,
        remote_version: str,
        fingerprint: str,
    ) -> tuple[str, bool]:
        now = utc_now()
        item_id = stable_item_id(tenant_key, source_token)
        existing = self.connection.execute(
            "SELECT metadata_hash, sync_status FROM documents WHERE source_token=?",
            (source_token,),
        ).fetchone()
        reactivated = bool(existing and existing["sync_status"] != "active")
        if existing is None:
            outcome = "new"
            self.connection.execute(
                """
                INSERT INTO documents(
                    source_token, folder_token, name, obj_type, remote_modified_time,
                    metadata_hash, sync_status, first_seen_at, last_seen_at,
                    last_changed_at, last_seen_run_id, tenant_key, root_token,
                    parent_folder_token, item_id, source_url, remote_version, missing_count
                ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    source_token,
                    parent_folder_token,
                    name,
                    obj_type,
                    remote_version,
                    fingerprint,
                    now,
                    now,
                    now,
                    run_id,
                    tenant_key,
                    root_token,
                    parent_folder_token,
                    item_id,
                    source_url,
                    remote_version,
                ),
            )
            if reactivated:
                # 删除任务尚未领取时，恢复应直接使它失效；已在运行的任务
                # 还会由 Worker 再检查文档状态，避免删掉刚恢复的内容。
                self.connection.execute(
                    """
                    UPDATE sync_jobs SET status='completed', completed_at=?,
                        updated_at=?, last_error='superseded by restore'
                    WHERE source_token=? AND job_type='mark_missing'
                      AND status='pending'
                    """,
                    (now, now, source_token),
                )
        else:
            changed = existing["metadata_hash"] != fingerprint or existing["sync_status"] != "active"
            outcome = "changed" if changed else "unchanged"
            self.connection.execute(
                """
                UPDATE documents SET folder_token=?, name=?, obj_type=?,
                    remote_modified_time=?, metadata_hash=?, sync_status='active',
                    last_seen_at=?, last_changed_at=CASE WHEN ? THEN ? ELSE last_changed_at END,
                    last_seen_run_id=?, tenant_key=?, root_token=?, parent_folder_token=?,
                    item_id=?, source_url=?, remote_version=?, missing_count=0,
                    soft_deleted_at=NULL, last_error=NULL
                WHERE source_token=?
                """,
                (
                    parent_folder_token,
                    name,
                    obj_type,
                    remote_version,
                    fingerprint,
                    now,
                    int(changed),
                    now,
                    run_id,
                    tenant_key,
                    root_token,
                    parent_folder_token,
                    item_id,
                    source_url,
                    remote_version,
                    source_token,
                ),
            )
        created = self.enqueue_job(
            source_token,
            "upsert",
            fingerprint,
            commit=False,
            revive_existing=reactivated,
        )
        self.connection.commit()
        return outcome, created

    def mark_unseen(self, run_id: int, tenant_key: str, root_token: str) -> tuple[int, int]:
        rows = self.connection.execute(
            """
            SELECT source_token, metadata_hash, missing_count
            FROM documents
            WHERE tenant_key=? AND root_token=? AND last_seen_run_id<>?
              AND sync_status NOT IN ('soft_deleted', 'ignored')
            """,
            (tenant_key, root_token, run_id),
        ).fetchall()
        missing = jobs = 0
        for row in rows:
            count = int(row["missing_count"] or 0) + 1
            status = "soft_deleted" if count >= 2 else "suspect_missing"
            soft_deleted_at = utc_now() if status == "soft_deleted" else None
            self.connection.execute(
                """
                UPDATE documents SET missing_count=?, sync_status=?, soft_deleted_at=?,
                    last_changed_at=? WHERE source_token=?
                """,
                (count, status, soft_deleted_at, utc_now(), row["source_token"]),
            )
            missing += 1
            if status == "soft_deleted":
                jobs += int(
                    self.enqueue_job(
                        row["source_token"],
                        "mark_missing",
                        str(row["metadata_hash"]),
                        commit=False,
                    )
                )
        self.connection.commit()
        return missing, jobs

    def enqueue_job(
        self,
        source_token: str,
        job_type: str,
        target_hash: str,
        *,
        not_before: str | None = None,
        commit: bool = True,
        revive_existing: bool = False,
    ) -> bool:
        now = utc_now()
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO sync_jobs(
                source_token, job_type, target_hash, status, attempts,
                created_at, updated_at, next_retry_at
            ) VALUES (?, ?, ?, 'pending', 0, ?, ?, ?)
            """,
            (source_token, job_type, target_hash, now, now, not_before or now),
        )
        if cursor.rowcount == 0 and revive_existing:
            cursor = self.connection.execute(
                """
                UPDATE sync_jobs SET status='pending', attempts=0,
                    updated_at=?, next_retry_at=?, completed_at=NULL,
                    last_error=NULL, lease_owner='', lease_until=NULL
                WHERE source_token=? AND job_type=? AND target_hash=?
                  AND status IN ('completed', 'failed')
                """,
                (now, not_before or now, source_token, job_type, target_hash),
            )
        if commit:
            self.connection.commit()
        return cursor.rowcount == 1

    def claim_job(self, owner: str, lease_seconds: int = 1800) -> SyncJob | None:
        now = utc_now()
        with self.immediate_transaction() as connection:
            connection.execute(
                """
                UPDATE sync_jobs SET status='pending', lease_owner='', lease_until=NULL,
                    updated_at=? WHERE status='running' AND lease_until IS NOT NULL
                    AND lease_until<=?
                """,
                (now, now),
            )
            row = connection.execute(
                """
                SELECT * FROM sync_jobs
                WHERE status='pending' AND (next_retry_at IS NULL OR next_retry_at<=?)
                ORDER BY created_at, id LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE sync_jobs SET status='running', attempts=attempts+1,
                    lease_owner=?, lease_until=?, updated_at=? WHERE id=?
                """,
                (owner, utc_after(lease_seconds), now, row["id"]),
            )
            return SyncJob(
                id=int(row["id"]),
                source_token=str(row["source_token"]),
                job_type=str(row["job_type"]),
                target_hash=str(row["target_hash"]),
                status="running",
                attempts=int(row["attempts"]) + 1,
                lease_owner=owner,
                lease_until=utc_after(lease_seconds),
            )

    def heartbeat(self, job_id: int, owner: str, lease_seconds: int = 1800) -> bool:
        cursor = self.connection.execute(
            """
            UPDATE sync_jobs SET lease_until=?, updated_at=?
            WHERE id=? AND status='running' AND lease_owner=?
            """,
            (utc_after(lease_seconds), utc_now(), job_id, owner),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def complete_job(self, job_id: int, owner: str) -> None:
        self.connection.execute(
            """
            UPDATE sync_jobs SET status='completed', completed_at=?, updated_at=?,
                lease_owner='', lease_until=NULL, last_error=NULL
            WHERE id=? AND lease_owner=?
            """,
            (utc_now(), utc_now(), job_id, owner),
        )
        self.connection.commit()

    def fail_job(self, job: SyncJob, owner: str, error: str, max_attempts: int) -> str:
        terminal = job.attempts >= max_attempts
        status = "failed" if terminal else "pending"
        delay = self.RETRY_DELAYS[min(max(0, job.attempts - 1), len(self.RETRY_DELAYS) - 1)]
        self.connection.execute(
            """
            UPDATE sync_jobs SET status=?, next_retry_at=?, updated_at=?, last_error=?,
                lease_owner='', lease_until=NULL WHERE id=? AND lease_owner=?
            """,
            (
                status,
                utc_after(delay),
                utc_now(),
                error[:2000],
                job.id,
                owner,
            ),
        )
        self.connection.execute(
            "UPDATE documents SET last_error=? WHERE source_token=?",
            (error[:2000], job.source_token),
        )
        self.connection.commit()
        return status

    def retry_job(self, job_id: int) -> bool:
        cursor = self.connection.execute(
            """
            UPDATE sync_jobs SET status='pending', attempts=0, next_retry_at=?,
                updated_at=?, last_error=NULL, lease_owner='', lease_until=NULL
            WHERE id=? AND status='failed'
            """,
            (utc_now(), utc_now(), job_id),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def get_document(self, source_token: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM documents WHERE source_token=?", (source_token,)
        ).fetchone()
        return dict(row) if row else None

    def get_document_by_item_id(self, item_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM documents WHERE item_id=?", (item_id,)
        ).fetchone()
        return dict(row) if row else None

    def mark_synced(
        self, source_token: str, content_hash: str, *, corpus_changed: bool = True
    ) -> None:
        self.connection.execute(
            """
            UPDATE documents SET content_hash=?, last_synced_at=?, last_error=NULL,
                sync_status='active' WHERE source_token=?
            """,
            (content_hash, utc_now(), source_token),
        )
        if corpus_changed:
            self.bump_corpus_revision(commit=False)
        self.connection.commit()

    def active_documents(self) -> list[dict[str, Any]]:
        """所有 sync_status=active 的文档，供漂移校验比对 Milvus 实际行数。"""
        rows = self.connection.execute(
            "SELECT source_token, item_id, name, metadata_hash FROM documents "
            "WHERE sync_status='active'"
        ).fetchall()
        return [dict(row) for row in rows]

    def clear_content_hash(self, source_token: str) -> bool:
        """清空 content_hash，使 Worker 的「内容未变」短路失效。

        只对仍保有哈希的行生效并返回 True；重复调用（任务还在排队时
        下一轮对账再校验）不会再次生效，避免修复动作重复计数。
        """
        cursor = self.connection.execute(
            "UPDATE documents SET content_hash='' WHERE source_token=? "
            "AND content_hash!=''",
            (source_token,),
        )
        changed = bool(cursor.rowcount)
        if changed:
            self.connection.commit()
        return changed

    def bump_corpus_revision(self, *, commit: bool = True) -> int:
        self.connection.execute(
            "UPDATE corpus_state SET revision=revision+1, updated_at=? WHERE singleton=1",
            (utc_now(),),
        )
        revision = int(
            self.connection.execute(
                "SELECT revision FROM corpus_state WHERE singleton=1"
            ).fetchone()[0]
        )
        if commit:
            self.connection.commit()
        return revision

    def corpus_revision(self) -> int:
        return int(
            self.connection.execute(
                "SELECT revision FROM corpus_state WHERE singleton=1"
            ).fetchone()[0]
        )

    def record_event(
        self, event_id: str, event_type: str, source_token: str, payload: dict[str, Any]
    ) -> bool:
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO sync_events(
                event_id, event_type, source_token, received_at, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event_type,
                source_token,
                utc_now(),
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def deactivate_document(self, source_token: str, reason: str = "event") -> bool:
        row = self.connection.execute(
            "SELECT metadata_hash, sync_status FROM documents WHERE source_token=?",
            (source_token,),
        ).fetchone()
        if row is None:
            return False
        if row["sync_status"] == "soft_deleted":
            return True
        self.connection.execute(
            """
            UPDATE documents SET sync_status='soft_deleted', soft_deleted_at=?,
                last_changed_at=?, last_error=? WHERE source_token=?
            """,
            (utc_now(), utc_now(), reason[:500], source_token),
        )
        self.enqueue_job(
            source_token,
            "mark_missing",
            str(row["metadata_hash"]),
            commit=False,
            revive_existing=True,
        )
        self.connection.commit()
        return True

    def list_documents(self, limit: int = 100, status: str = "") -> list[dict[str, Any]]:
        sql = "SELECT * FROM documents"
        params: list[Any] = []
        if status:
            sql += " WHERE sync_status=?"
            params.append(status)
        sql += " ORDER BY last_changed_at DESC LIMIT ?"
        params.append(max(1, min(limit, 500)))
        return [dict(row) for row in self.connection.execute(sql, params)]

    def list_jobs(self, limit: int = 100, status: str = "") -> list[dict[str, Any]]:
        sql = "SELECT * FROM sync_jobs"
        params: list[Any] = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, min(limit, 500)))
        return [dict(row) for row in self.connection.execute(sql, params)]

    def status(self) -> dict[str, Any]:
        document_counts = {
            str(row["sync_status"]): int(row["count"])
            for row in self.connection.execute(
                "SELECT sync_status, COUNT(*) count FROM documents GROUP BY sync_status"
            )
        }
        job_counts = {
            str(row["status"]): int(row["count"])
            for row in self.connection.execute(
                "SELECT status, COUNT(*) count FROM sync_jobs GROUP BY status"
            )
        }
        sources = [dict(row) for row in self.connection.execute("SELECT * FROM sources")]
        latest_run = self.connection.execute(
            "SELECT * FROM sync_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return {
            "documents": document_counts,
            "jobs": job_counts,
            "sources": sources,
            "corpus_revision": self.corpus_revision(),
            "latest_run": dict(latest_run) if latest_run else None,
        }

    def purge_expired_soft_deletes(self, grace_days: int) -> list[dict[str, Any]]:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=max(1, grace_days))
        ).isoformat(timespec="seconds")
        rows = self.connection.execute(
            """
            SELECT * FROM documents WHERE sync_status='soft_deleted'
              AND soft_deleted_at IS NOT NULL AND soft_deleted_at<=?
              AND NOT EXISTS (
                SELECT 1 FROM sync_jobs
                WHERE sync_jobs.source_token=documents.source_token
                  AND sync_jobs.job_type='mark_missing'
                  AND sync_jobs.status IN ('pending', 'running')
              )
            """,
            (cutoff,),
        ).fetchall()
        return [dict(row) for row in rows]

    def mark_purged(self, source_token: str) -> None:
        """保留最小审计记录，但清掉正文哈希和可恢复状态。"""

        self.connection.execute(
            """
            UPDATE documents SET sync_status='purged', content_hash=NULL,
                last_changed_at=? WHERE source_token=? AND sync_status='soft_deleted'
            """,
            (utc_now(), source_token),
        )
        self.connection.commit()
