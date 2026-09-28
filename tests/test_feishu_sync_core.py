from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.feishu_sync.adapters import normalize_docx_blocks, normalize_pdf
from app.feishu_sync.client import FeishuClient
from app.feishu_sync.events import handle_event_payload
from app.feishu_sync.store import SyncStore, metadata_hash, stable_item_id


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self.payload

    def iter_content(self, chunk_size):
        del chunk_size
        yield b"complete-pdf"


class FakeSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if url.endswith("tenant_access_token/internal"):
            return FakeResponse(
                {"code": 0, "tenant_access_token": "token", "expire": 7200}
            )
        folder = kwargs.get("params", {}).get("folder_token")
        files = {
            "root": [
                {"token": "child", "name": "子目录", "type": "folder"},
                {"token": "doc-1", "name": "周报", "type": "docx"},
            ],
            "child": [
                {"token": "pdf-1", "name": "论文.pdf", "type": "file"}
            ],
        }
        return FakeResponse({"code": 0, "data": {"files": files[folder], "has_more": False}})


def test_client_walks_nested_folders_and_caches_token():
    session = FakeSession()
    client = FeishuClient("app", "secret", session=session)

    items = list(client.walk_folder("root"))
    second = list(client.walk_folder("root"))

    assert [(parent, item["token"]) for parent, item in items] == [
        ("root", "child"),
        ("root", "doc-1"),
        ("child", "pdf-1"),
    ]
    assert len(second) == 3
    token_calls = [call for call in session.calls if call[1].endswith("tenant_access_token/internal")]
    assert len(token_calls) == 1


def test_docx_blocks_become_normalized_document():
    document = normalize_docx_blocks(
        document_id="doc-1",
        item_id="1234567890abcdef",
        title="测试周报",
        source_url="https://example.feishu.cn/docx/doc-1",
        blocks=[
            {
                "block_id": "heading",
                "heading1": {"elements": [{"text_run": {"content": "进展"}}]},
            },
            {
                "block_id": "paragraph",
                "text": {"elements": [{"text_run": {"content": "完成自动同步"}}]},
            },
        ],
    )

    assert document.title == "测试周报"
    assert [block.type for block in document.blocks] == ["heading", "text"]
    assert document.blocks[1].text == "完成自动同步"
    assert len(document.content_hash) == 64


def test_docx_table_empty_and_long_text_are_preserved():
    long_text = "长段落" * 2000
    document = normalize_docx_blocks(
        document_id="doc-2",
        item_id="abcdef1234567890",
        title="复杂文档",
        source_url="",
        blocks=[
            {"block_id": "table", "table": {"cells": ["A", "B"]}},
            {
                "block_id": "long",
                "text": {"elements": [{"text_run": {"content": long_text}}]},
            },
        ],
    )
    empty = normalize_docx_blocks(
        document_id="empty",
        item_id="0000000000000000",
        title="空文档",
        source_url="",
        blocks=[],
    )

    assert document.blocks[0].type == "table"
    assert "<table>" in document.blocks[0].html
    assert document.blocks[1].text == long_text
    assert empty.blocks[0].text == "[空文档]"


def test_synthetic_pdf_is_extracted():
    path = Path(__file__).resolve().parents[1] / "examples" / "synthetic_document.pdf"
    document = normalize_pdf(
        document_id="pdf-1",
        item_id="1234567890abcdef",
        title="合成 PDF",
        source_url="",
        path=path,
        content_hash="hash",
    )
    assert "blue-orchid-2026" in document.blocks[0].text


def test_interrupted_download_never_replaces_destination(tmp_path):
    class BrokenResponse(FakeResponse):
        def iter_content(self, chunk_size):
            del chunk_size
            yield b"partial"
            raise OSError("connection reset")

    class BrokenSession(FakeSession):
        def request(self, method, url, **kwargs):
            if url.endswith("tenant_access_token/internal"):
                return FakeResponse(
                    {"code": 0, "tenant_access_token": "token", "expire": 7200}
                )
            return BrokenResponse({})

    destination = tmp_path / "paper.pdf"
    destination.write_bytes(b"old-complete-file")
    client = FeishuClient("app", "secret", session=BrokenSession())

    with pytest.raises(OSError):
        client.download_file("file-1", destination)

    assert destination.read_bytes() == b"old-complete-file"
    assert not (tmp_path / "paper.pdf.part").exists()


def test_store_migrates_legacy_database_and_deduplicates_jobs(tmp_path):
    path = tmp_path / "sync.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE documents (
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
            last_seen_run_id INTEGER NOT NULL
        );
        CREATE TABLE sync_jobs (
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
        """
    )
    connection.close()

    with SyncStore(path) as store:
        run_id = store.start_run()
        kwargs = dict(
            run_id=run_id,
            tenant_key="tenant",
            root_token="root",
            parent_folder_token="root",
            source_token="doc-1",
            name="周报",
            obj_type="docx",
            source_url="https://example/docx/doc-1",
            remote_version="100",
            fingerprint="hash-1",
        )
        first = store.record_document(**kwargs)
        second = store.record_document(**kwargs)

        columns = {
            row["name"] for row in store.connection.execute("PRAGMA table_info(documents)")
        }
        jobs = store.list_jobs()

    assert first == ("new", True)
    assert second == ("unchanged", False)
    assert "tenant_key" in columns
    assert len(jobs) == 1


def test_job_lease_retry_and_completion(tmp_path):
    with SyncStore(tmp_path / "sync.db") as store:
        run_id = store.start_run()
        store.record_document(
            run_id=run_id,
            tenant_key="tenant",
            root_token="root",
            parent_folder_token="root",
            source_token="doc-1",
            name="周报",
            obj_type="docx",
            source_url="",
            remote_version="100",
            fingerprint="hash-1",
        )
        job = store.claim_job("worker-1")
        assert job is not None
        assert job.attempts == 1
        assert store.heartbeat(job.id, "worker-1") is True
        assert store.fail_job(job, "worker-1", "temporary", max_attempts=5) == "pending"

        store.connection.execute(
            "UPDATE sync_jobs SET next_retry_at='2000-01-01T00:00:00+00:00' WHERE id=?",
            (job.id,),
        )
        store.connection.commit()
        retried = store.claim_job("worker-2")
        assert retried is not None
        store.complete_job(retried.id, "worker-2")
        assert store.list_jobs()[0]["status"] == "completed"


def test_item_id_and_metadata_hash_are_stable():
    item = {"token": "doc-1", "name": "周报", "type": "docx", "modified_time": "1"}
    assert stable_item_id("tenant", "doc-1") == stable_item_id("tenant", "doc-1")
    assert len(stable_item_id("tenant", "doc-1")) == 16
    assert metadata_hash("root", "parent", item) == metadata_hash("root", "parent", dict(item))


def test_events_are_deduplicated_and_delete_creates_one_job(tmp_path):
    with SyncStore(tmp_path / "sync.db") as store:
        run_id = store.start_run()
        store.record_document(
            run_id=run_id,
            tenant_key="tenant",
            root_token="root",
            parent_folder_token="root",
            source_token="doc-1",
            name="周报",
            obj_type="docx",
            source_url="",
            remote_version="1",
            fingerprint="meta-1",
        )
        payload = {
            "header": {"event_id": "event-1", "event_type": "drive.file.deleted_v1"},
            "event": {"file_token": "doc-1"},
        }

        assert handle_event_payload(store, payload) is True
        assert handle_event_payload(store, payload) is False
        assert store.get_document("doc-1")["sync_status"] == "soft_deleted"
        jobs = store.list_jobs()
        assert len([job for job in jobs if job["job_type"] == "mark_missing"]) == 1


def test_deleted_document_can_be_restored_with_same_remote_hash(tmp_path):
    with SyncStore(tmp_path / "sync.db") as store:
        kwargs = dict(
            tenant_key="tenant",
            root_token="root",
            parent_folder_token="root",
            source_token="doc-1",
            name="周报",
            obj_type="docx",
            source_url="https://example/docx/doc-1",
            remote_version="1",
            fingerprint="same-hash",
        )
        first_run = store.start_run()
        store.record_document(run_id=first_run, **kwargs)
        upsert = store.claim_job("worker")
        store.complete_job(upsert.id, "worker")

        assert store.deactivate_document("doc-1") is True
        missing = next(job for job in store.list_jobs() if job["job_type"] == "mark_missing")
        claimed_missing = store.claim_job("worker")
        assert claimed_missing.id == missing["id"]
        store.complete_job(claimed_missing.id, "worker")

        second_run = store.start_run()
        outcome, queued = store.record_document(run_id=second_run, **kwargs)

        assert outcome == "changed"
        assert queued is True
        restored_upsert = next(
            job for job in store.list_jobs() if job["job_type"] == "upsert"
        )
        assert restored_upsert["status"] == "pending"
