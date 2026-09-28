from __future__ import annotations

import argparse
import logging
import threading
import time

from .events import WebSocketEventRunner
from .service import ReconcileService
from .settings import FeishuSyncSettings
from .worker import SyncWorker


logger = logging.getLogger(__name__)


def _settings() -> FeishuSyncSettings:
    settings = FeishuSyncSettings.from_app_settings()
    settings.validate()
    return settings


def _drift_checker():
    """对账时顺带校验台账与 Milvus 是否脱节（向量库被重建/误删的场景）。"""

    from .ingestion import IngestionService

    return IngestionService().drifted_items


def run_sync_once(settings: FeishuSyncSettings) -> dict:
    service = ReconcileService(settings, drift_checker=_drift_checker())
    try:
        result = service.reconcile().as_dict()
    finally:
        service.close()
    print(
        "同步扫描完成："
        f"发现 {result['discovered']}，新增 {result['new']}，"
        f"变化 {result['changed']}，未变化 {result['unchanged']}，"
        f"缺失 {result['missing']}，新建任务 {result['jobs_created']}"
    )
    return result


def run_worker_once(settings: FeishuSyncSettings) -> int:
    worker = SyncWorker(settings)
    processed = 0
    try:
        while worker.run_once():
            processed += 1
        purged = worker.cleanup_expired()
    finally:
        worker.close()
    print(f"Worker 本轮处理 {processed} 个任务，清理 {purged} 个过期软删除快照")
    return processed


def run_watch(settings: FeishuSyncSettings) -> None:
    wake_reconcile = threading.Event()

    def reconcile_loop() -> None:
        while True:
            try:
                run_sync_once(settings)
            except Exception:
                logger.exception("飞书定时对账失败，将在下一周期重试")
            wake_reconcile.wait(settings.reconcile_interval_seconds)
            wake_reconcile.clear()

    worker = SyncWorker(settings)
    threading.Thread(target=reconcile_loop, daemon=True, name="feishu-reconcile").start()
    threading.Thread(
        target=worker.run_forever, daemon=True, name="feishu-worker"
    ).start()

    if settings.event_mode == "websocket":
        try:
            WebSocketEventRunner(settings, wake_reconcile.set).run_forever()
        except Exception:
            # 长连接权限、订阅或 SDK 出问题时不让 Worker 跟着退出；
            # 定时递归对账仍会补齐漏掉的变化。
            logger.exception("飞书 WebSocket 不可用，已降级为定时对账")
            while True:
                time.sleep(3600)
    else:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            worker.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="飞书知识库增量同步")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sync = subparsers.add_parser("sync", help="扫描飞书共享文件夹并生成任务")
    sync.add_argument("--once", action="store_true", help="扫描一次后退出")
    worker = subparsers.add_parser("worker", help="处理待入库任务")
    worker.add_argument("--once", action="store_true", help="清空当前可执行任务后退出")
    subparsers.add_parser("watch", help="启动事件、定时对账与 Worker")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args(argv)
    settings = _settings()
    if args.command == "sync":
        run_sync_once(settings)
    elif args.command == "worker":
        run_worker_once(settings)
    else:
        run_watch(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
