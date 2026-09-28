from __future__ import annotations

import app.api as api
from app.feishu_sync.store import SyncStore


def _seed(path):
    with SyncStore(path) as store:
        store.ensure_source("private-tenant", "private-root")
        run_id = store.start_run()
        store.record_document(
            run_id=run_id,
            tenant_key="private-tenant",
            root_token="private-root",
            parent_folder_token="private-parent",
            source_token="private-token",
            name="公开标题",
            obj_type="docx",
            source_url="https://example.feishu.cn/docx/example",
            remote_version="1",
            fingerprint="hash-1",
        )
        job = store.claim_job("worker")
        store.fail_job(job, "worker", "权限不足 Log ID: abc", max_attempts=1)


def test_admin_sync_status_and_lists_hide_tokens(monkeypatch, tmp_path):
    path = tmp_path / "sync.db"
    _seed(path)
    monkeypatch.setattr(api, "_open_sync_store", lambda: SyncStore(path))

    status = api.admin_feishu_sync_status()
    documents = api.admin_feishu_sync_documents(status="", limit=100)["documents"]
    jobs = api.admin_feishu_sync_jobs(status="", limit=100)["jobs"]

    assert status["jobs"]["failed"] == 1
    assert documents[0]["name"] == "公开标题"
    assert "source_token" not in documents[0]
    assert "tenant_key" not in documents[0]
    assert "folder_token" not in documents[0]
    assert "source_token" not in jobs[0]
    assert "tenant_key" not in status["sources"][0]
    assert "root_token" not in status["sources"][0]


def test_admin_failed_job_can_be_retried(monkeypatch, tmp_path):
    path = tmp_path / "sync.db"
    _seed(path)
    monkeypatch.setattr(api, "_open_sync_store", lambda: SyncStore(path))
    with SyncStore(path) as store:
        job_id = store.list_jobs()[0]["id"]

    response = api.admin_feishu_sync_retry(job_id)

    assert response["status"] == "pending"
    with SyncStore(path) as store:
        assert store.list_jobs()[0]["status"] == "pending"
