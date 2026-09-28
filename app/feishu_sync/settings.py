from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import PROJECT_DIR, Settings, get_settings


@dataclass(frozen=True)
class FeishuSyncSettings:
    app_id: str
    app_secret: str
    folder_tokens: tuple[str, ...]
    enabled: bool
    event_mode: str
    reconcile_interval_seconds: int
    db_path: Path
    allowed_types: frozenset[str]
    delete_grace_days: int
    worker_max_attempts: int
    pdf_parser: str
    download_dir: Path

    @classmethod
    def from_app_settings(cls, settings: Settings | None = None) -> "FeishuSyncSettings":
        settings = settings or get_settings()
        raw_db = Path(settings.feishu_sync_db_path)
        raw_download = Path(settings.feishu_sync_download_dir)
        return cls(
            app_id=settings.feishu_app_id,
            app_secret=settings.feishu_app_secret,
            folder_tokens=tuple(
                token.strip()
                for token in settings.feishu_folder_tokens.replace(";", ",").split(",")
                if token.strip()
            ),
            enabled=settings.feishu_sync_enabled,
            event_mode=settings.feishu_event_mode,
            reconcile_interval_seconds=max(60, settings.feishu_reconcile_interval_seconds),
            db_path=raw_db if raw_db.is_absolute() else PROJECT_DIR / raw_db,
            allowed_types=frozenset(
                item.strip().lower()
                for item in settings.feishu_allowed_types.split(",")
                if item.strip()
            ),
            delete_grace_days=max(1, settings.feishu_delete_grace_days),
            worker_max_attempts=max(1, settings.feishu_worker_max_attempts),
            pdf_parser=settings.feishu_pdf_parser.lower(),
            download_dir=(
                raw_download if raw_download.is_absolute() else PROJECT_DIR / raw_download
            ),
        )

    def validate(self) -> None:
        missing = []
        if not self.app_id:
            missing.append("FEISHU_APP_ID")
        if not self.app_secret:
            missing.append("FEISHU_APP_SECRET")
        if not self.folder_tokens:
            missing.append("FEISHU_FOLDER_TOKENS")
        if missing:
            raise ValueError("缺少飞书同步配置：" + ", ".join(missing))
        if self.event_mode not in {"websocket", "polling"}:
            raise ValueError("FEISHU_EVENT_MODE 只能是 websocket 或 polling")
        if self.pdf_parser not in {"auto", "mineru", "pypdf"}:
            raise ValueError("FEISHU_PDF_PARSER 只能是 auto、mineru 或 pypdf")
