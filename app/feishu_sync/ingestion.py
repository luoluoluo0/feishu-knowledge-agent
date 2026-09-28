from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from filelock import FileLock

from app.config import PROJECT_DIR, Settings, get_settings

from .models import NormalizedDocument

if TYPE_CHECKING:
    from app.milvus_store import MilvusStore


SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_chunks_v2  # noqa: E402


PARENTS_FILE = PROJECT_DIR / "data" / "processed" / "parents.json"
CHUNKS_FILE = PROJECT_DIR / "data" / "processed" / "chunks_v2.jsonl"
INGEST_LOCK = PROJECT_DIR / "data" / "runtime" / "ingest.lock"
DELETED_DIR = PROJECT_DIR / "data" / "runtime" / "feishu_sync" / "deleted"
SNAPSHOT_DIR = PROJECT_DIR / "data" / "runtime" / "feishu_sync" / "snapshots"


class SyncIngestionError(RuntimeError):
    pass


def chunk_document(document: NormalizedDocument) -> tuple[list[dict], list[dict]]:
    blocks = [block.as_ingestion_block() for block in document.blocks]
    groups = build_chunks_v2.group_by_section(blocks)
    parents: list[dict] = []
    children: list[dict] = []
    child_counter = 0
    parent_counter = 0
    meta = {"title": document.title, "reader": "", "doi": ""}
    for section, section_blocks in groups:
        section_children = build_chunks_v2.build_children(
            section, section_blocks, document.item_id, child_counter
        )
        if not section_children:
            continue
        child_counter += len(section_children)
        section_parents = build_chunks_v2.make_parents(
            document.item_id,
            section,
            section_children,
            meta,
            parent_counter,
        )
        parent_counter += len(section_parents)
        is_reference = build_chunks_v2.is_reference_section(section, section_blocks)
        for chunk in section_parents + section_children:
            chunk["title"] = document.title
            chunk["reader"] = ""
            chunk["doi"] = ""
            chunk["is_reference"] = is_reference
        parents.extend(section_parents)
        children.extend(section_children)
    if not children:
        raise SyncIngestionError("规范化文档没有生成任何可检索子块")
    return parents, children


def _load_parents() -> list[dict]:
    if not PARENTS_FILE.exists():
        return []
    with PARENTS_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, list) else []


def _load_children() -> list[dict]:
    if not CHUNKS_FILE.exists():
        return []
    rows = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


class IngestionService:
    """把规范化文档提交到现有 Milvus，同时维护父子块本地快照。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store_factory: Callable[[Settings], Any] | None = None,
        lock_path: Path = INGEST_LOCK,
    ) -> None:
        self.settings = settings or get_settings()
        if store_factory is None:
            from app.milvus_store import MilvusStore

            store_factory = MilvusStore
        self.store_factory = store_factory
        self.lock_path = lock_path

    def ingest(self, document: NormalizedDocument) -> dict:
        parents, children = chunk_document(document)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.lock_path), timeout=1800):
            return self._commit(document, parents, children)

    def _commit(
        self,
        document: NormalizedDocument,
        parents: list[dict],
        children: list[dict],
    ) -> dict:
        store = self.store_factory(self.settings)
        store.ensure_collection()
        old_parents_all = _load_parents()
        old_children_all = _load_children()
        old_parents = [p for p in old_parents_all if p.get("item_id") == document.item_id]
        old_children = [c for c in old_children_all if c.get("item_id") == document.item_id]
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _atomic_json(
            SNAPSHOT_DIR / f"{document.item_id}-{stamp}.json",
            {"parents": old_parents, "children": old_children},
        )
        new_parents_all = [
            p for p in old_parents_all if p.get("item_id") != document.item_id
        ] + parents
        new_children_all = [
            c for c in old_children_all if c.get("item_id") != document.item_id
        ] + children

        self._delete_from_milvus(store, document.item_id)
        try:
            inserted = store.insert_chunks(children)
            if inserted != len(children):
                raise SyncIngestionError(
                    f"Milvus 实际写入 {inserted} 条，预期 {len(children)} 条"
                )
            count = self._wait_for_count(store, document.item_id, len(children))
            if count != len(children):
                raise SyncIngestionError(
                    f"Milvus 查回 {count} 条，预期 {len(children)} 条"
                )
            _atomic_json(PARENTS_FILE, new_parents_all)
            _atomic_jsonl(CHUNKS_FILE, new_children_all)
        except Exception as exc:
            self._delete_from_milvus(store, document.item_id)
            if old_children:
                store.insert_chunks(old_children)
                restored = self._wait_for_count(
                    store, document.item_id, len(old_children)
                )
                if restored != len(old_children):
                    raise SyncIngestionError(
                        f"新版本失败且旧版本恢复不完整：{restored}/{len(old_children)}"
                    ) from exc
            _atomic_json(PARENTS_FILE, old_parents_all)
            _atomic_jsonl(CHUNKS_FILE, old_children_all)
            raise SyncIngestionError(f"入库失败，旧版本已恢复：{exc}") from exc
        return {
            "item_id": document.item_id,
            "inserted": inserted,
            "parents": len(parents),
            "content_hash": document.content_hash,
        }

    def remove(self, item_id: str) -> dict:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.lock_path), timeout=1800):
            store = self.store_factory(self.settings)
            store.ensure_collection()
            parents_all = _load_parents()
            children_all = _load_children()
            old_parents = [p for p in parents_all if p.get("item_id") == item_id]
            old_children = [c for c in children_all if c.get("item_id") == item_id]
            DELETED_DIR.mkdir(parents=True, exist_ok=True)
            _atomic_json(
                DELETED_DIR / f"{item_id}.json",
                {"parents": old_parents, "children": old_children},
            )
            deleted = self._delete_from_milvus(store, item_id)
            _atomic_json(
                PARENTS_FILE,
                [p for p in parents_all if p.get("item_id") != item_id],
            )
            _atomic_jsonl(
                CHUNKS_FILE,
                [c for c in children_all if c.get("item_id") != item_id],
            )
            return {
                "item_id": item_id,
                "deleted": deleted,
                "local_parents": len(old_parents),
                "local_children": len(old_children),
            }

    @staticmethod
    def purge_deleted_snapshot(item_id: str) -> bool:
        """删除超过宽限期的本地软删除快照。"""

        path = DELETED_DIR / f"{item_id}.json"
        if not path.exists():
            return False
        path.unlink()
        return True

    @staticmethod
    def _count_item(store: Any, item_id: str) -> int:
        rows = store.client.query(
            collection_name=store.collection,
            filter=f'item_id == "{_escape_expr_value(item_id)}"',
            output_fields=["chunk_id"],
            limit=16384,
        )
        return len(rows)

    @staticmethod
    def _delete_from_milvus(store: Any, item_id: str) -> int:
        before = IngestionService._count_item(store, item_id)
        if before:
            store.client.delete(
                collection_name=store.collection,
                filter=f'item_id == "{_escape_expr_value(item_id)}"',
            )
            store.client.flush(store.collection)
            remaining = IngestionService._wait_for_count(store, item_id, 0)
            if remaining:
                raise SyncIngestionError(
                    f"Milvus 删除后仍查到 {remaining} 条旧数据，停止替换"
                )
        return before

    @staticmethod
    def _wait_for_count(
        store: Any, item_id: str, expected: int, timeout: float = 15.0
    ) -> int:
        deadline = time.monotonic() + timeout
        count = IngestionService._count_item(store, item_id)
        while count != expected and time.monotonic() < deadline:
            time.sleep(0.1)
            count = IngestionService._count_item(store, item_id)
        return count


def _escape_expr_value(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')
