from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .client import FeishuClient
from .settings import FeishuSyncSettings
from .store import SyncStore, metadata_hash


@dataclass
class ReconcileResult:
    discovered: int = 0
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    missing: int = 0
    jobs_created: int = 0
    ignored: int = 0

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


class ReconcileService:
    def __init__(
        self,
        settings: FeishuSyncSettings,
        *,
        client: FeishuClient | None = None,
        store: SyncStore | None = None,
    ) -> None:
        settings.validate()
        self.settings = settings
        self.client = client or FeishuClient(settings.app_id, settings.app_secret)
        self.store = store or SyncStore(settings.db_path)
        self._owns_store = store is None

    def close(self) -> None:
        if self._owns_store:
            self.store.close()

    def reconcile(self) -> ReconcileResult:
        run_id = self.store.start_run()
        result = ReconcileResult()
        try:
            self.client.authenticate()
            errors: list[str] = []
            for root_token in self.settings.folder_tokens:
                self.store.ensure_source(self.client.tenant_key, root_token)
                try:
                    self._scan_root(run_id, root_token, result)
                except Exception as exc:
                    self.store.set_source_status(
                        self.client.tenant_key, root_token, "degraded", str(exc)
                    )
                    errors.append(f"{root_token}: {exc}")
                else:
                    self.store.set_source_status(
                        self.client.tenant_key, root_token, "healthy"
                    )
            self.store.finish_run(run_id, result.as_dict(), "; ".join(errors))
            return result
        except Exception as exc:
            self.store.finish_run(run_id, result.as_dict(), str(exc))
            raise

    def _scan_root(
        self, run_id: int, root_token: str, result: ReconcileResult
    ) -> None:
        for parent_token, item in self.client.walk_folder(root_token):
            item_type = str(item.get("type") or "unknown").lower()
            if item_type == "folder":
                continue
            name = str(item.get("name") or "未命名")
            source_type = "pdf" if item_type == "file" and name.lower().endswith(".pdf") else item_type
            if source_type not in self.settings.allowed_types:
                result.ignored += 1
                continue
            source_token = self.client.item_token(item)
            if not source_token:
                result.ignored += 1
                continue
            version = str(item.get("modified_time") or item.get("modified_at") or "")
            url = str(item.get("url") or item.get("shortcut_info", {}).get("target_token") or "")
            fingerprint = metadata_hash(root_token, parent_token, item)
            outcome, created = self.store.record_document(
                run_id=run_id,
                tenant_key=self.client.tenant_key,
                root_token=root_token,
                parent_folder_token=parent_token,
                source_token=source_token,
                name=name,
                obj_type=source_type,
                source_url=url,
                remote_version=version,
                fingerprint=fingerprint,
            )
            result.discovered += 1
            setattr(result, outcome, getattr(result, outcome) + 1)
            result.jobs_created += int(created)
        missing, jobs = self.store.mark_unseen(
            run_id, self.client.tenant_key, root_token
        )
        result.missing += missing
        result.jobs_created += jobs
