from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import app.feishu_sync.ingestion as ingestion_module
from app.feishu_sync.ingestion import IngestionService, SyncIngestionError
from app.feishu_sync.models import NormalizedBlock, NormalizedDocument
from app.feishu_sync.settings import FeishuSyncSettings
from app.feishu_sync.store import SyncStore
from app.feishu_sync.worker import SyncWorker


class FakeMilvusClient:
    def __init__(self):
        self.rows = []

    def query(self, collection_name, filter, output_fields, limit):
        del collection_name, output_fields, limit
        item_id = filter.split('"')[1]
        return [row for row in self.rows if row.get("item_id") == item_id]

    def delete(self, collection_name, filter):
        del collection_name
        item_id = filter.split('"')[1]
        self.rows = [row for row in self.rows if row.get("item_id") != item_id]

    def flush(self, collection_name):
        del collection_name


class FakeMilvusStore:
    def __init__(self, client, fail=False):
        self.client = client
        self.collection = "test"
        self.fail = fail

    def ensure_collection(self):
        return None

    def insert_chunks(self, chunks):
        if self.fail:
            self.fail = False
            raise RuntimeError("insert failed")
        self.client.rows.extend(dict(chunk) for chunk in chunks)
        return len(chunks)


def _document(text="新内容"):
    return NormalizedDocument(
        document_id="doc-1",
        item_id="1234567890abcdef",
        title="周报",
        source_type="docx",
        source_url="https://example/docx/doc-1",
        content_hash="content-hash",
        blocks=(
            NormalizedBlock("h1", "heading", text="进展"),
            NormalizedBlock("p1", "text", text=text),
        ),
    )


def _redirect_files(monkeypatch, tmp_path):
    parents = tmp_path / "parents.json"
    children = tmp_path / "chunks.jsonl"
    deleted = tmp_path / "deleted"
    parents.write_text("[]", encoding="utf-8")
    children.write_text("", encoding="utf-8")
    monkeypatch.setattr(ingestion_module, "PARENTS_FILE", parents)
    monkeypatch.setattr(ingestion_module, "CHUNKS_FILE", children)
    monkeypatch.setattr(ingestion_module, "DELETED_DIR", deleted)
    return parents, children


def test_ingestion_writes_children_and_parents(monkeypatch, tmp_path):
    parents_path, children_path = _redirect_files(monkeypatch, tmp_path)
    client = FakeMilvusClient()
    fake = FakeMilvusStore(client)
    service = IngestionService(
        store_factory=lambda settings: fake,
        lock_path=tmp_path / "ingest.lock",
    )

    report = service.ingest(_document())

    parents = json.loads(parents_path.read_text(encoding="utf-8"))
    children = [
        json.loads(line)
        for line in children_path.read_text(encoding="utf-8").splitlines()
    ]
    assert report["inserted"] == len(children) == len(client.rows)
    assert parents[0]["item_id"] == "1234567890abcdef"
    assert children[0]["parent_id"] == parents[0]["chunk_id"]


def test_ingestion_restores_old_version_on_failure(monkeypatch, tmp_path):
    parents_path, children_path = _redirect_files(monkeypatch, tmp_path)
    old_parent = {"chunk_id": "1234567890abcdef_p0001", "item_id": "1234567890abcdef"}
    old_child = {
        "chunk_id": "1234567890abcdef_c0001",
        "item_id": "1234567890abcdef",
        "text": "旧内容",
    }
    parents_path.write_text(json.dumps([old_parent], ensure_ascii=False), encoding="utf-8")
    children_path.write_text(json.dumps(old_child, ensure_ascii=False) + "\n", encoding="utf-8")
    client = FakeMilvusClient()
    client.rows = [dict(old_child)]
    fake = FakeMilvusStore(client, fail=True)
    service = IngestionService(
        store_factory=lambda settings: fake,
        lock_path=tmp_path / "ingest.lock",
    )

    with pytest.raises(SyncIngestionError):
        service.ingest(_document())

    assert client.rows == [old_child]
    assert json.loads(parents_path.read_text(encoding="utf-8")) == [old_parent]
    restored = json.loads(children_path.read_text(encoding="utf-8").strip())
    assert restored == old_child


class FakeFeishuClient:
    def list_document_blocks(self, document_id):
        assert document_id == "doc-1"
        return [
            {
                "block_id": "paragraph",
                "text": {"elements": [{"text_run": {"content": "自动同步成功"}}]},
            }
        ]


class FakeIngestion:
    def __init__(self):
        self.documents = []
        self.removed = []

    def ingest(self, document):
        self.documents.append(document)
        return {"inserted": 1}

    def remove(self, item_id):
        self.removed.append(item_id)
        return {"deleted": 1}


def _settings(tmp_path):
    return FeishuSyncSettings(
        app_id="app",
        app_secret="secret",
        folder_tokens=("root",),
        enabled=True,
        event_mode="polling",
        reconcile_interval_seconds=3600,
        db_path=tmp_path / "sync.db",
        allowed_types=frozenset({"docx", "pdf"}),
        delete_grace_days=7,
        worker_max_attempts=5,
        pdf_parser="pypdf",
        download_dir=tmp_path / "downloads",
    )


def test_worker_completes_docx_job_and_updates_corpus_revision(tmp_path):
    settings = _settings(tmp_path)
    fake_ingestion = FakeIngestion()
    with SyncStore(settings.db_path) as store:
        run_id = store.start_run()
        store.record_document(
            run_id=run_id,
            tenant_key="tenant",
            root_token="root",
            parent_folder_token="root",
            source_token="doc-1",
            name="周报",
            obj_type="docx",
            source_url="https://example/docx/doc-1",
            remote_version="1",
            fingerprint="meta-1",
        )
        worker = SyncWorker(
            settings,
            client=FakeFeishuClient(),
            store=store,
            ingestion=fake_ingestion,
            owner="worker",
        )

        assert worker.run_once() is True

        assert store.list_jobs()[0]["status"] == "completed"
        assert store.corpus_revision() == 1
        assert len(fake_ingestion.documents) == 1
        assert store.get_document("doc-1")["content_hash"]


def test_worker_purges_snapshot_after_delete_grace_period(tmp_path):
    settings = _settings(tmp_path)
    fake_ingestion = FakeIngestion()
    fake_ingestion.purged = []
    fake_ingestion.purge_deleted_snapshot = fake_ingestion.purged.append
    with SyncStore(settings.db_path) as store:
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
        item_id = store.get_document("doc-1")["item_id"]
        store.deactivate_document("doc-1")
        old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat(
            timespec="seconds"
        )
        store.connection.execute(
            "UPDATE documents SET soft_deleted_at=? WHERE source_token='doc-1'",
            (old,),
        )
        store.connection.execute(
            "UPDATE sync_jobs SET status='completed' WHERE job_type='mark_missing'"
        )
        store.connection.commit()
        worker = SyncWorker(
            settings,
            client=FakeFeishuClient(),
            store=store,
            ingestion=fake_ingestion,
            owner="worker",
        )

        assert worker.cleanup_expired() == 1
        assert store.get_document("doc-1")["sync_status"] == "purged"
        assert fake_ingestion.purged == [item_id]
