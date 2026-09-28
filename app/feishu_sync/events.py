from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from .settings import FeishuSyncSettings
from .store import SyncStore, utc_after


logger = logging.getLogger(__name__)

DELETE_EVENTS = {"drive.file.deleted_v1", "drive.file.trashed_v1"}
CHANGE_EVENTS = {
    "drive.file.edit_v1",
    "drive.file.title_updated_v1",
    "drive.file.created_in_folder_v1",
}


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def handle_event_payload(
    store: SyncStore,
    payload: dict[str, Any],
    *,
    trigger_reconcile: Callable[[], None] | None = None,
) -> bool:
    header = _as_dict(payload.get("header"))
    event = _as_dict(payload.get("event"))
    event_id = str(header.get("event_id") or payload.get("event_id") or "")
    event_type = str(header.get("event_type") or payload.get("event_type") or "")
    source_token = str(
        event.get("file_token")
        or event.get("token")
        or _as_dict(event.get("file")).get("token")
        or ""
    )
    if not event_id or not event_type:
        return False
    if not store.record_event(event_id, event_type, source_token, payload):
        return False
    if event_type in DELETE_EVENTS and source_token:
        store.deactivate_document(source_token, event_type)
        return True
    if event_type in CHANGE_EVENTS:
        if source_token and store.get_document(source_token):
            bucket = int(time.time() // 60)
            store.enqueue_job(
                source_token,
                "upsert",
                f"event:{bucket}",
                not_before=utc_after(60),
            )
        if trigger_reconcile:
            trigger_reconcile()
        return True
    if trigger_reconcile:
        trigger_reconcile()
    return True


class WebSocketEventRunner:
    """飞书官方 SDK 长连接封装；SDK 缺失时给出可操作错误。"""

    def __init__(
        self,
        settings: FeishuSyncSettings,
        trigger_reconcile: Callable[[], None] | None = None,
    ) -> None:
        self.settings = settings
        self.trigger_reconcile = trigger_reconcile

    def run_forever(self) -> None:
        try:
            import lark_oapi as lark
        except ImportError as exc:
            raise RuntimeError(
                "WebSocket 模式需要 lark-oapi，请先安装 requirements.txt"
            ) from exc

        def callback(data) -> None:
            # CustomizedEvent 不是普通 dict；官方 SDK 的 JSON.marshal 才会
            # 把 header/event 完整序列化出来。
            payload = _as_dict(lark.JSON.marshal(data))
            with SyncStore(self.settings.db_path) as store:
                handle_event_payload(
                    store,
                    payload,
                    trigger_reconcile=self.trigger_reconcile,
                )

        handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_customized_event("drive.file.edit_v1", callback)
            .register_p2_customized_event("drive.file.title_updated_v1", callback)
            .register_p2_customized_event("drive.file.deleted_v1", callback)
            .register_p2_customized_event("drive.file.trashed_v1", callback)
            .register_p2_customized_event("drive.file.created_in_folder_v1", callback)
            .build()
        )
        client = lark.ws.Client(
            self.settings.app_id,
            self.settings.app_secret,
            event_handler=handler,
            log_level=lark.LogLevel.INFO,
        )
        client.start()
