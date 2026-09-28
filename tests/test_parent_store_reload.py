from __future__ import annotations

import json
import os
import time

from app.milvus_store import ParentStore


def test_parent_store_reloads_when_file_changes(tmp_path):
    path = tmp_path / "parents.json"
    path.write_text(
        json.dumps([{"chunk_id": "p1", "text": "旧内容"}], ensure_ascii=False),
        encoding="utf-8",
    )
    store = ParentStore(path)
    assert store.get("p1")["text"] == "旧内容"

    path.write_text(
        json.dumps([{"chunk_id": "p1", "text": "新内容"}], ensure_ascii=False),
        encoding="utf-8",
    )
    future = time.time_ns() + 2_000_000_000
    os.utime(path, ns=(future, future))

    assert store.get("p1")["text"] == "新内容"
