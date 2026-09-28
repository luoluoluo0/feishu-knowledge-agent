from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Iterator

import requests


API_BASE = "https://open.feishu.cn/open-apis"


class FeishuAPIError(RuntimeError):
    def __init__(
        self,
        action: str,
        message: str,
        *,
        code: int | str = "",
        status_code: int = 0,
        log_id: str = "",
    ) -> None:
        self.action = action
        self.code = code
        self.status_code = status_code
        self.log_id = log_id
        detail = f"{action}失败：{message}"
        if code != "":
            detail += f"（code={code}）"
        if log_id:
            detail += f"，Log ID: {log_id}"
        super().__init__(detail)


class FeishuClient:
    """飞书只读 REST 客户端；令牌只存在内存，不进入日志。"""

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        *,
        session: requests.Session | None = None,
        timeout: float = 20.0,
        max_retries: int = 3,
    ) -> None:
        self.app_id = app_id
        self._app_secret = app_secret
        self._session = session or requests.Session()
        self._timeout = timeout
        self._max_retries = max(1, max_retries)
        self._token = ""
        self._token_expires_at = 0.0

    @property
    def tenant_key(self) -> str:
        return hashlib.sha256(self.app_id.encode("utf-8")).hexdigest()[:12]

    def authenticate(self, *, force: bool = False) -> str:
        if not force and self._token and time.monotonic() < self._token_expires_at:
            return self._token
        payload = self._request(
            "POST",
            "/auth/v3/tenant_access_token/internal",
            action="获取 tenant_access_token",
            authenticated=False,
            json={"app_id": self.app_id, "app_secret": self._app_secret},
        )
        token = payload.get("tenant_access_token")
        if not isinstance(token, str) or not token:
            raise FeishuAPIError("获取 tenant_access_token", "响应缺少访问令牌")
        expire = int(payload.get("expire") or 7200)
        self._token = token
        self._token_expires_at = time.monotonic() + max(60, expire - 300)
        return token

    def list_folder_items(self, folder_token: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page_token = ""
        while True:
            params: dict[str, Any] = {"folder_token": folder_token, "page_size": 200}
            if page_token:
                params["page_token"] = page_token
            payload = self._request(
                "GET", "/drive/v1/files", action="列出飞书文件夹", params=params
            )
            data = payload.get("data") or {}
            files = data.get("files") or []
            if not isinstance(files, list):
                raise FeishuAPIError("列出飞书文件夹", "data.files 不是列表")
            items.extend(item for item in files if isinstance(item, dict))
            if not data.get("has_more"):
                return items
            page_token = str(data.get("next_page_token") or data.get("page_token") or "")
            if not page_token:
                raise FeishuAPIError("列出飞书文件夹", "响应声称有下一页但没有游标")

    def walk_folder(self, root_token: str) -> Iterator[tuple[str, dict[str, Any]]]:
        pending = [root_token]
        visited: set[str] = set()
        while pending:
            folder_token = pending.pop()
            if folder_token in visited:
                continue
            visited.add(folder_token)
            for item in self.list_folder_items(folder_token):
                yield folder_token, item
                if item.get("type") == "folder":
                    token = self.item_token(item)
                    if token and token not in visited:
                        pending.append(token)

    def list_document_blocks(self, document_id: str) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        page_token = ""
        while True:
            params: dict[str, Any] = {"page_size": 500, "document_revision_id": -1}
            if page_token:
                params["page_token"] = page_token
            payload = self._request(
                "GET",
                f"/docx/v1/documents/{document_id}/blocks",
                action="读取飞书文档块",
                params=params,
            )
            data = payload.get("data") or {}
            items = data.get("items") or []
            if not isinstance(items, list):
                raise FeishuAPIError("读取飞书文档块", "data.items 不是列表")
            blocks.extend(item for item in items if isinstance(item, dict))
            if not data.get("has_more"):
                return blocks
            page_token = str(data.get("page_token") or "")
            if not page_token:
                raise FeishuAPIError("读取飞书文档块", "响应缺少下一页游标")

    def download_file(self, file_token: str, destination: Path) -> tuple[str, int]:
        response = self._request_raw(
            "GET",
            f"/drive/v1/files/{file_token}/download",
            action="下载飞书文件",
            stream=True,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            if not size:
                raise FeishuAPIError("下载飞书文件", "返回了空文件")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return digest.hexdigest(), size

    def subscribe(self, file_token: str, *, file_type: str = "docx", event_type: str = "") -> None:
        body: dict[str, Any] = {"file_type": file_type}
        if event_type:
            body["event_type"] = event_type
        self._request(
            "POST",
            f"/drive/v1/files/{file_token}/subscribe",
            action="订阅飞书文档事件",
            json=body,
        )

    @staticmethod
    def item_token(item: dict[str, Any]) -> str:
        return str(item.get("token") or item.get("file_token") or "")

    def _request(
        self,
        method: str,
        path: str,
        *,
        action: str,
        authenticated: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        response = self._request_raw(
            method, path, action=action, authenticated=authenticated, **kwargs
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise FeishuAPIError(action, "响应不是 JSON", status_code=response.status_code) from exc
        if not isinstance(payload, dict):
            raise FeishuAPIError(action, "响应 JSON 不是对象")
        code = payload.get("code", 0)
        if code != 0:
            raise FeishuAPIError(
                action,
                str(payload.get("msg") or payload.get("message") or "未知错误"),
                code=code,
                status_code=response.status_code,
                log_id=response.headers.get("X-Tt-Logid", ""),
            )
        return payload

    def _request_raw(
        self,
        method: str,
        path: str,
        *,
        action: str,
        authenticated: bool = True,
        **kwargs: Any,
    ) -> requests.Response:
        headers = dict(kwargs.pop("headers", {}) or {})
        if authenticated:
            headers["Authorization"] = f"Bearer {self.authenticate()}"
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._session.request(
                    method,
                    f"{API_BASE}{path}",
                    headers=headers,
                    timeout=self._timeout,
                    **kwargs,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt + 1 < self._max_retries:
                    time.sleep(0.25 * (2**attempt))
                    continue
                raise FeishuAPIError(action, f"网络请求失败：{exc}") from exc
            if response.status_code == 401 and authenticated and attempt == 0:
                headers["Authorization"] = f"Bearer {self.authenticate(force=True)}"
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt + 1 < self._max_retries:
                    retry_after = float(response.headers.get("Retry-After") or 0.25 * (2**attempt))
                    time.sleep(min(retry_after, 5.0))
                    continue
            if response.status_code >= 400:
                try:
                    payload = response.json()
                except ValueError:
                    payload = {}
                raise FeishuAPIError(
                    action,
                    str(payload.get("msg") or payload.get("message") or f"HTTP {response.status_code}"),
                    code=payload.get("code", ""),
                    status_code=response.status_code,
                    log_id=response.headers.get("X-Tt-Logid", ""),
                )
            return response
        raise FeishuAPIError(action, f"请求失败：{last_error or '未知错误'}")
