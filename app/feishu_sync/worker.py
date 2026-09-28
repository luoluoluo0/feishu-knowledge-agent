from __future__ import annotations

import hashlib
import logging
import os
import socket
import shutil
import threading
import time
import re
from pathlib import Path
from uuid import uuid4

from .adapters import normalize_docx_blocks, normalize_pdf
from .client import FeishuClient
from .ingestion import IngestionService
from .models import NormalizedBlock, NormalizedDocument, SyncJob
from .settings import FeishuSyncSettings
from .store import SyncStore


logger = logging.getLogger(__name__)


def safe_filename(value: str, fallback: str) -> str:
    forbidden = '<>:"/\\|?*'
    name = Path(value).name.strip()
    for char in forbidden:
        name = name.replace(char, "_")
    return name.rstrip(". ") or fallback


class SyncWorker:
    def __init__(
        self,
        settings: FeishuSyncSettings,
        *,
        client: FeishuClient | None = None,
        store: SyncStore | None = None,
        ingestion: IngestionService | None = None,
        owner: str | None = None,
    ) -> None:
        self.settings = settings
        self.client = client or FeishuClient(settings.app_id, settings.app_secret)
        self.store = store or SyncStore(settings.db_path)
        self.ingestion = ingestion or IngestionService()
        self.owner = owner or f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"
        self._owns_store = store is None

    def close(self) -> None:
        if self._owns_store:
            self.store.close()

    def run_once(self) -> bool:
        job = self.store.claim_job(self.owner)
        if job is None:
            return False
        stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat_loop, args=(job.id, stop), daemon=True
        )
        heartbeat.start()
        try:
            self._process(job)
            self.store.complete_job(job.id, self.owner)
            return True
        except Exception as exc:
            status = self.store.fail_job(
                job, self.owner, f"{type(exc).__name__}: {exc}", self.settings.worker_max_attempts
            )
            logger.exception("飞书同步任务 %s 处理失败，状态=%s", job.id, status)
            return True
        finally:
            stop.set()
            heartbeat.join(timeout=1)

    def run_forever(self, poll_seconds: float = 2.0) -> None:
        next_cleanup = 0.0
        try:
            while True:
                if time.monotonic() >= next_cleanup:
                    self.cleanup_expired()
                    next_cleanup = time.monotonic() + 3600
                if not self.run_once():
                    time.sleep(max(0.2, poll_seconds))
        finally:
            self.close()

    def _heartbeat_loop(self, job_id: int, stop: threading.Event) -> None:
        # 心跳使用独立 SQLite 连接，避免与主线程的事务共享 connection。
        with SyncStore(self.settings.db_path) as heartbeat_store:
            while not stop.wait(60):
                heartbeat_store.heartbeat(job_id, self.owner)

    def _process(self, job: SyncJob) -> None:
        record = self.store.get_document(job.source_token)
        if not record:
            raise RuntimeError("任务对应的文档记录不存在")
        if job.job_type == "mark_missing":
            if record.get("sync_status") != "soft_deleted":
                return
            report = self.ingestion.remove(str(record["item_id"]))
            if report.get("deleted") or report.get("local_parents") or report.get("local_children"):
                self.store.bump_corpus_revision()
            return
        document = self._fetch_document(record)
        if record.get("content_hash") == document.content_hash:
            self.store.mark_synced(job.source_token, document.content_hash, corpus_changed=False)
            return
        self.ingestion.ingest(document)
        self.store.mark_synced(job.source_token, document.content_hash)

    def cleanup_expired(self) -> int:
        """清理超过软删除宽限期的快照和已下载原文件。"""

        purged = 0
        for record in self.store.purge_expired_soft_deletes(
            self.settings.delete_grace_days
        ):
            item_id = str(record.get("item_id") or "")
            # 目录只能是本系统生成的 16 位哈希，避免把异常台账值当路径。
            if not re.fullmatch(r"[0-9a-f]{16}", item_id):
                logger.warning("跳过非法 item_id 的清理：%r", item_id)
                continue
            self.ingestion.purge_deleted_snapshot(item_id)
            download_dir = (self.settings.download_dir / item_id).resolve()
            root = self.settings.download_dir.resolve()
            if download_dir.parent == root and download_dir.exists():
                shutil.rmtree(download_dir)
            self.store.mark_purged(str(record["source_token"]))
            purged += 1
        return purged

    def _fetch_document(self, record: dict) -> NormalizedDocument:
        source_type = str(record["obj_type"])
        if source_type == "docx":
            blocks = self.client.list_document_blocks(str(record["source_token"]))
            return normalize_docx_blocks(
                document_id=str(record["source_token"]),
                item_id=str(record["item_id"]),
                title=str(record["name"]),
                source_url=str(record.get("source_url") or ""),
                blocks=blocks,
            )
        if source_type == "pdf":
            filename = safe_filename(
                str(record["name"]), f"{record['item_id']}.pdf"
            )
            destination = self.settings.download_dir / str(record["item_id"]) / filename
            content_hash, _ = self.client.download_file(
                str(record["source_token"]), destination
            )
            if self.settings.pdf_parser == "mineru":
                return self._normalize_with_mineru(record, destination, content_hash)
            if self.settings.pdf_parser == "auto":
                try:
                    return self._normalize_with_mineru(record, destination, content_hash)
                except Exception:
                    logger.info("MinerU 不可用，回退到 pypdf：%s", filename, exc_info=True)
            return normalize_pdf(
                document_id=str(record["source_token"]),
                item_id=str(record["item_id"]),
                title=str(record["name"]),
                source_url=str(record.get("source_url") or ""),
                path=destination,
                content_hash=content_hash,
            )
        raise ValueError(f"暂不支持的飞书文档类型：{source_type}")

    @staticmethod
    def _normalize_with_mineru(
        record: dict, path: Path, content_hash: str
    ) -> NormalizedDocument:
        import sys

        scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import ingest_paper
        import parse_mineru

        json_path = ingest_paper.run_mineru_parse(path)
        parse_mineru.parse_one(json_path, path, str(record["item_id"]))
        block_path = parse_mineru.BLOCK_DIR / f"{record['item_id']}.jsonl"
        blocks = []
        with block_path.open("r", encoding="utf-8") as file:
            for index, line in enumerate(file, start=1):
                payload = __import__("json").loads(line)
                blocks.append(
                    NormalizedBlock(
                        block_id=str(payload.get("block_id") or f"block-{index}"),
                        type=str(payload.get("block_type") or "text"),
                        text=str(payload.get("text") or ""),
                        html=str(payload.get("html") or ""),
                        page=int(payload.get("page") or 0),
                        page_end=int(payload.get("page_end") or payload.get("page") or 0),
                        image_path=str(payload.get("image_path") or ""),
                        label=str(payload.get("label") or ""),
                    )
                )
        return NormalizedDocument(
            document_id=str(record["source_token"]),
            item_id=str(record["item_id"]),
            title=str(record["name"]),
            source_type="pdf",
            source_url=str(record.get("source_url") or ""),
            content_hash=content_hash,
            blocks=tuple(blocks),
            local_path=path,
        )
