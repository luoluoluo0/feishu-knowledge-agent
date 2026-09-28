from pathlib import Path

from scripts.check_feishu_sync_ledger import (
    FeishuSyncLedger,
    metadata_fingerprint,
)


def _item(token: str, name: str, modified_time: str = "100"):
    return {
        "token": token,
        "name": name,
        "type": "docx",
        "modified_time": modified_time,
    }


def test_metadata_fingerprint_is_stable_and_detects_changes():
    item = _item("doc-1", "周报")
    assert metadata_fingerprint("folder-1", item) == metadata_fingerprint(
        "folder-1", dict(item)
    )
    assert metadata_fingerprint("folder-1", item) != metadata_fingerprint(
        "folder-1", _item("doc-1", "周报", "101")
    )


def test_first_scan_is_new_and_second_scan_is_unchanged(tmp_path):
    path = tmp_path / "sync.db"
    items = [_item("doc-1", "周报"), _item("doc-2", "月报")]

    with FeishuSyncLedger(path) as ledger:
        first_run = ledger.start_run("2026-01-01T00:00:00+00:00")
        first = ledger.record_folder(first_run, "folder-1", items, "time-1")
        first.missing, missing_jobs = ledger.mark_missing(
            first_run, ("folder-1",), "time-1"
        )
        first.jobs_created += missing_jobs
        ledger.finish_run(first_run, first, "time-1")

        second_run = ledger.start_run("2026-01-01T01:00:00+00:00")
        second = ledger.record_folder(second_run, "folder-1", items, "time-2")
        second.missing, missing_jobs = ledger.mark_missing(
            second_run, ("folder-1",), "time-2"
        )
        second.jobs_created += missing_jobs
        ledger.finish_run(second_run, second, "time-2")

    assert (first.new, first.unchanged, first.missing) == (2, 0, 0)
    assert (second.new, second.unchanged, second.missing) == (0, 2, 0)
    assert first.jobs_created == 2
    assert second.jobs_created == 0


def test_changed_and_missing_documents_are_detected(tmp_path):
    path = tmp_path / "sync.db"

    with FeishuSyncLedger(path) as ledger:
        first_run = ledger.start_run("start-1")
        first = ledger.record_folder(
            first_run,
            "folder-1",
            [_item("doc-1", "周报", "100"), _item("doc-2", "月报", "100")],
            "time-1",
        )
        ledger.finish_run(first_run, first, "time-1")

        second_run = ledger.start_run("start-2")
        second = ledger.record_folder(
            second_run,
            "folder-1",
            [_item("doc-1", "周报", "101")],
            "time-2",
        )
        second.missing, missing_jobs = ledger.mark_missing(
            second_run, ("folder-1",), "time-2"
        )
        second.jobs_created += missing_jobs
        ledger.finish_run(second_run, second, "time-2")

        rows = ledger.connection.execute(
            "SELECT source_token, sync_status FROM documents ORDER BY source_token"
        ).fetchall()
        jobs = ledger.connection.execute(
            """
            SELECT source_token, job_type, status
            FROM sync_jobs
            ORDER BY id
            """
        ).fetchall()

    assert second.changed == 1
    assert second.missing == 1
    assert [(row["source_token"], row["sync_status"]) for row in rows] == [
        ("doc-1", "active"),
        ("doc-2", "missing"),
    ]
    assert [(row["source_token"], row["job_type"]) for row in jobs] == [
        ("doc-1", "upsert"),
        ("doc-2", "upsert"),
        ("doc-1", "upsert"),
        ("doc-2", "mark_missing"),
    ]


def test_unchanged_scan_backfills_job_but_never_duplicates_it(tmp_path):
    path = tmp_path / "sync.db"
    item = _item("doc-1", "周报")

    with FeishuSyncLedger(path) as ledger:
        run_id = ledger.start_run("start-1")
        first = ledger.record_folder(run_id, "folder-1", [item], "time-1")
        ledger.finish_run(run_id, first, "time-1")

        second_run = ledger.start_run("start-2")
        second = ledger.record_folder(second_run, "folder-1", [item], "time-2")
        ledger.finish_run(second_run, second, "time-2")

        count = ledger.connection.execute(
            "SELECT COUNT(*) FROM sync_jobs"
        ).fetchone()[0]

    assert first.jobs_created == 1
    assert second.jobs_created == 0
    assert count == 1


def test_database_parent_is_created(tmp_path):
    path = tmp_path / "nested" / "sync.db"
    with FeishuSyncLedger(path):
        pass
    assert path.exists()
