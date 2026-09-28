"""增量入库纯逻辑测试。不触 Milvus / MinerU / API，纯文件与函数。"""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from ingest_paper import (  # noqa: E402
    load_manifest,
    merge_chunks_lines,
    merge_parents_by_item,
    needs_ingest,
    next_item_id,
    qc_parent_child_consistent,
    save_manifest,
    upsert_csv_row,
)


def make_parent(item_id, number, child_ids):
    return {
        "chunk_id": f"{item_id}_p{number:04d}",
        "chunk_type": "parent",
        "item_id": item_id,
        "child_ids": child_ids,
        "text": "父块内容",
    }


def make_child(item_id, number, parent_id):
    return {
        "chunk_id": f"{item_id}_c{number:04d}",
        "chunk_type": "child",
        "item_id": item_id,
        "parent_id": parent_id,
        "text": "子块内容",
    }


class TestNeedsIngest:
    def test_new_paper(self):
        needed, reason = needs_ingest(None, "hash1", False)
        assert needed is True
        assert "新文献" in reason

    def test_same_hash_skips(self):
        entry = {"pdf_sha256": "hash1"}
        needed, reason = needs_ingest(entry, "hash1", False)
        assert needed is False
        assert "幂等" in reason

    def test_changed_hash_ingests(self):
        entry = {"pdf_sha256": "hash1"}
        needed, _ = needs_ingest(entry, "hash2", False)
        assert needed is True

    def test_force_overrides_same_hash(self):
        entry = {"pdf_sha256": "hash1"}
        needed, _ = needs_ingest(entry, "hash1", True)
        assert needed is True


class TestManifest:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "manifest.jsonl"
        entries = {"045": {"item_id": "045", "pdf_sha256": "h1"}}
        save_manifest(path, entries)
        assert load_manifest(path) == entries

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_manifest(tmp_path / "none.jsonl") == {}

    def test_last_write_wins(self, tmp_path):
        path = tmp_path / "manifest.jsonl"
        path.write_text(
            json.dumps({"item_id": "045", "v": 1}) + "\n"
            + json.dumps({"item_id": "045", "v": 2}) + "\n",
            encoding="utf-8",
        )
        assert load_manifest(path)["045"]["v"] == 2


class TestMergeParents:
    def test_replaces_target_keeps_others(self):
        old = [make_parent("044", 1, []), make_parent("045", 1, [])]
        new = [make_parent("045", 1, ["045_c0001"])]
        merged = merge_parents_by_item(old, "045", new)
        ids = [p["chunk_id"] for p in merged]
        assert "044_p0001" in ids
        assert merged.count(new[0]) == 1
        # 旧 045 父块被替换（同 id 只出现一次）
        assert ids.count("045_p0001") == 1

    def test_other_papers_untouched(self):
        old = [make_parent("044", 1, [])]
        merged = merge_parents_by_item(old, "045", [make_parent("045", 1, [])])
        assert merged[0] == old[0]


class TestMergeChunksLines:
    def test_replaces_target_lines(self):
        old_lines = [
            json.dumps(make_child("044", 1, "044_p0001"), ensure_ascii=False),
            json.dumps(make_child("045", 1, "045_p0001"), ensure_ascii=False),
        ]
        new_children = [make_child("045", 1, "045_p0001"), make_child("045", 2, "045_p0001")]
        merged = merge_chunks_lines(old_lines, "045", new_children)
        assert len(merged) == 3
        assert sum(1 for line in merged if '"item_id": "045"' in line) == 2

    def test_other_lines_untouched(self):
        keep = json.dumps(make_child("044", 1, "044_p0001"), ensure_ascii=False)
        merged = merge_chunks_lines([keep], "045", [make_child("045", 1, "045_p0001")])
        assert merged[0] == keep


class TestUpsertCsvRow:
    def test_append_new(self):
        rows = []
        rows, action = upsert_csv_row(rows, "120", {"title": "新论文", "status": "ready"})
        assert action == "append"
        assert rows[0]["item_id"] == "120"
        assert rows[0]["title"] == "新论文"

    def test_update_existing(self):
        rows = [{"item_id": "045", "title": "旧标题", "status": ""}]
        rows, action = upsert_csv_row(rows, "045", {"title": "新标题", "status": "ready"})
        assert action == "update"
        assert len(rows) == 1
        assert rows[0]["title"] == "新标题"
        assert rows[0]["status"] == "ready"


class TestQcParentChild:
    def test_consistent_passes(self):
        parents = [make_parent("045", 1, ["045_c0001", "045_c0002"])]
        children = [make_child("045", 1, "045_p0001"), make_child("045", 2, "045_p0001")]
        ok, _ = qc_parent_child_consistent(parents, children)
        assert ok

    def test_dangling_child_reference_fails(self):
        parents = [make_parent("045", 1, ["045_c0001", "045_c9999"])]
        children = [make_child("045", 1, "045_p0001")]
        ok, detail = qc_parent_child_consistent(parents, children)
        assert not ok
        assert "045_c9999" in detail

    def test_orphan_parent_id_fails(self):
        parents = [make_parent("045", 1, ["045_c0001"])]
        children = [make_child("045", 1, "045_p9999")]
        ok, detail = qc_parent_child_consistent(parents, children)
        assert not ok
        assert "parent_id" in detail

    def test_foreign_parent_fails(self):
        parents = [make_parent("045", 1, []), make_parent("044", 1, [])]
        children = [make_child("045", 1, "045_p0001")]
        ok, _ = qc_parent_child_consistent(parents, children)
        assert not ok


class TestNextItemId:
    def test_next_number(self):
        assert next_item_id(["001", "044", "102"]) == "103"

    def test_empty(self):
        assert next_item_id([]) == "001"

    def test_ignores_non_numeric(self):
        assert next_item_id(["001", "demo"]) == "002"
