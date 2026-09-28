from __future__ import annotations

import json

from app.feishu_sync.ingestion import IngestionService, local_child_counts
from app.feishu_sync.service import ReconcileService
from app.feishu_sync.settings import FeishuSyncSettings
from app.feishu_sync.store import SyncStore, metadata_hash


DOC_ITEM = {
    "token": "doc-1",
    "name": "周报",
    "type": "docx",
    "url": "https://example/docx/doc-1",
    "modified_time": "1",
}


def doc_fingerprint() -> str:
    return metadata_hash("root", "root", DOC_ITEM)


class FakeFeishuClient:
    """扫描时发现台账里已有的那篇文档，保持 active 状态。"""

    tenant_key = "tenant"

    def authenticate(self, *, force: bool = False) -> str:
        return "token"

    @staticmethod
    def item_token(item: dict) -> str:
        return str(item.get("token") or "")

    def walk_folder(self, root_token: str):
        yield root_token, DOC_ITEM


def _settings(tmp_path) -> FeishuSyncSettings:
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


def _seed_settled_doc(store: SyncStore) -> None:
    """造出「台账认为已同步、任务已完成」的稳态。"""
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
        fingerprint=doc_fingerprint(),
    )
    job = store.claim_job("owner")
    assert job is not None
    store.complete_job(job.id, "owner")
    store.mark_synced("doc-1", "content-1")


def _pending_jobs(store: SyncStore) -> list[dict]:
    rows = store.connection.execute(
        "SELECT source_token, job_type, status FROM sync_jobs WHERE status='pending'"
    ).fetchall()
    return [dict(row) for row in rows]


def _reconcile(tmp_path, checker) -> dict:
    settings = _settings(tmp_path)
    with SyncStore(settings.db_path) as store:
        _seed_settled_doc(store)
        service = ReconcileService(
            settings, client=FakeFeishuClient(), store=store, drift_checker=checker
        )
        result = service.reconcile().as_dict()
        result["_pending"] = _pending_jobs(store)
        result["_hash"] = store.get_document("doc-1").get("content_hash")
        return result


def test_drift_repair_clears_hash_and_revives_job(tmp_path):
    result = _reconcile(tmp_path, lambda ids: set(ids))

    assert result["changed"] == 1
    assert result["jobs_created"] == 1
    assert result["_hash"] == ""
    assert result["_pending"] == [
        {"source_token": "doc-1", "job_type": "upsert", "status": "pending"}
    ]


def test_drift_repair_is_idempotent_while_job_pending(tmp_path):
    settings = _settings(tmp_path)
    with SyncStore(settings.db_path) as store:
        _seed_settled_doc(store)
        service = ReconcileService(
            settings,
            client=FakeFeishuClient(),
            store=store,
            drift_checker=lambda ids: set(ids),
        )
        first = service.reconcile().as_dict()
        second = service.reconcile().as_dict()

    assert first["changed"] == 1
    assert second["changed"] == 0
    assert second["jobs_created"] == 0


def test_no_drift_leads_to_no_repair(tmp_path):
    result = _reconcile(tmp_path, lambda ids: set())

    assert result["changed"] == 0
    assert result["_hash"] == "content-1"
    assert result["_pending"] == []


def test_unreachable_milvus_skips_verification(tmp_path):
    def broken_checker(ids):
        raise RuntimeError("Milvus 不可达")

    result = _reconcile(tmp_path, broken_checker)

    assert result["changed"] == 0
    assert result["_hash"] == "content-1"
    assert result["_pending"] == []


class FakeMilvusStore:
    """只实现 drifted_items 用到的最小接口。"""

    collection = "fake"

    def __init__(self, counts: dict[str, int]):
        self.counts = counts
        self.client = self

    def ensure_collection(self, recreate: bool = False) -> None:
        pass

    def query(self, *, filter: str, **kwargs):
        item_id = filter.split('"')[1]
        return [{"chunk_id": "row"}] * self.counts.get(item_id, 0)


def test_drifted_items_compares_mirror_with_milvus(tmp_path, monkeypatch):
    mirror = tmp_path / "chunks_v2.jsonl"
    rows = [
        {"item_id": "aaa", "chunk_id": "aaa_c0001"},
        {"item_id": "aaa", "chunk_id": "aaa_c0002"},
        {"item_id": "bbb", "chunk_id": "bbb_c0001"},
    ]
    mirror.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.feishu_sync.ingestion.CHUNKS_FILE", mirror
    )

    assert local_child_counts() == {"aaa": 2, "bbb": 1}

    store = FakeMilvusStore({"aaa": 0, "bbb": 1})  # aaa 的向量丢了
    service = IngestionService(
        settings=object(), store_factory=lambda _settings: store
    )

    assert service.drifted_items(["aaa", "bbb", "ccc"]) == {"aaa"}
