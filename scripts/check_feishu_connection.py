"""只读检查飞书应用凭证、文件夹权限与在线文档读取能力。

这个脚本是旁路探针，不会修改现有 Agent、数据库或飞书文档。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from pypdf import PdfReader


FEISHU_API_BASE = "https://open.feishu.cn/open-apis"


class FeishuProbeError(RuntimeError):
    """飞书连接检查失败，消息中不得包含应用密钥或访问令牌。"""


@dataclass(frozen=True)
class FeishuProbeSettings:
    app_id: str
    app_secret: str
    folder_tokens: tuple[str, ...]


def split_folder_tokens(value: str) -> tuple[str, ...]:
    normalized = value.replace(";", ",").replace("\n", ",")
    return tuple(token.strip() for token in normalized.split(",") if token.strip())


def load_settings(env_path: Path | None = None) -> FeishuProbeSettings:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(env_path or project_root / ".env", override=False)

    app_id = os.getenv("FEISHU_APP_ID", "").strip()
    app_secret = os.getenv("FEISHU_APP_SECRET", "").strip()
    folder_tokens = split_folder_tokens(os.getenv("FEISHU_FOLDER_TOKENS", ""))

    missing = []
    if not app_id:
        missing.append("FEISHU_APP_ID")
    if not app_secret:
        missing.append("FEISHU_APP_SECRET")
    if not folder_tokens:
        missing.append("FEISHU_FOLDER_TOKENS")
    if missing:
        raise FeishuProbeError(f".env 缺少配置：{', '.join(missing)}")

    return FeishuProbeSettings(app_id, app_secret, folder_tokens)


class FeishuReadOnlyClient:
    def __init__(
        self,
        app_id: str,
        app_secret: str,
        *,
        session: requests.Session | None = None,
        timeout: float = 15.0,
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._session = session or requests.Session()
        self._timeout = timeout
        self._tenant_access_token: str | None = None

    def authenticate(self) -> None:
        try:
            response = self._session.post(
                f"{FEISHU_API_BASE}/auth/v3/tenant_access_token/internal",
                json={"app_id": self._app_id, "app_secret": self._app_secret},
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise FeishuProbeError(f"获取 tenant_access_token 时网络请求失败：{exc}") from exc

        payload = self._decode_response(response, "获取 tenant_access_token")
        token = payload.get("tenant_access_token")
        if not isinstance(token, str) or not token:
            raise FeishuProbeError("飞书返回成功，但响应中没有 tenant_access_token")
        self._tenant_access_token = token

    def list_folder_items(self, folder_token: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page_token: str | None = None

        while True:
            params: dict[str, Any] = {
                "folder_token": folder_token,
                "page_size": 200,
            }
            if page_token:
                params["page_token"] = page_token

            payload = self._get_json(
                "/drive/v1/files",
                params=params,
                action="列出飞书文件夹",
            )
            data = payload.get("data") or {}
            page_items = data.get("files") or []
            if not isinstance(page_items, list):
                raise FeishuProbeError("飞书文件列表响应格式异常：data.files 不是列表")
            items.extend(item for item in page_items if isinstance(item, dict))

            if not data.get("has_more"):
                break
            page_token = data.get("next_page_token") or data.get("page_token")
            if not page_token:
                raise FeishuProbeError("飞书提示还有下一页，但没有返回 page_token")

        return items

    def get_docx_raw_content(self, document_id: str) -> str:
        payload = self._get_json(
            f"/docx/v1/documents/{document_id}/raw_content",
            params={"lang": 0},
            action="读取飞书在线文档",
        )
        content = (payload.get("data") or {}).get("content")
        if not isinstance(content, str):
            raise FeishuProbeError("飞书文档响应格式异常：data.content 不是字符串")
        return content

    def download_file(self, file_token: str, destination: Path) -> tuple[Path, str, int]:
        """下载普通云盘文件，使用临时文件避免失败时留下半截文件。"""

        if not self._tenant_access_token:
            raise FeishuProbeError("尚未认证，请先调用 authenticate()")
        try:
            response = self._session.get(
                f"{FEISHU_API_BASE}/drive/v1/files/{file_token}/download",
                headers={"Authorization": f"Bearer {self._tenant_access_token}"},
                timeout=self._timeout,
                stream=True,
            )
        except requests.RequestException as exc:
            raise FeishuProbeError(f"下载飞书文件时网络请求失败：{exc}") from exc

        if response.status_code >= 400:
            self._decode_response(response, "下载飞书文件")

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.name}.part")
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if size == 0:
                raise FeishuProbeError("飞书返回了空文件，未覆盖目标文件")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

        return destination, digest.hexdigest(), size

    def _get_json(
        self,
        path: str,
        *,
        params: dict[str, Any],
        action: str,
    ) -> dict[str, Any]:
        if not self._tenant_access_token:
            raise FeishuProbeError("尚未认证，请先调用 authenticate()")
        try:
            response = self._session.get(
                f"{FEISHU_API_BASE}{path}",
                params=params,
                headers={"Authorization": f"Bearer {self._tenant_access_token}"},
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise FeishuProbeError(f"{action}时网络请求失败：{exc}") from exc
        return self._decode_response(response, action)

    @staticmethod
    def _decode_response(response: requests.Response, action: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise FeishuProbeError(
                f"{action}失败：HTTP {response.status_code}，响应不是 JSON"
            ) from exc

        if not isinstance(payload, dict):
            raise FeishuProbeError(f"{action}失败：响应 JSON 不是对象")

        code = payload.get("code", 0)
        if response.status_code >= 400 or code != 0:
            message = payload.get("msg") or payload.get("message") or "未知错误"
            request_id = response.headers.get("X-Tt-Logid", "")
            suffix = f"，Log ID: {request_id}" if request_id else ""
            raise FeishuProbeError(
                f"{action}失败：HTTP {response.status_code}，code={code}，msg={message}{suffix}"
            )
        return payload


def _item_token(item: dict[str, Any]) -> str:
    return str(item.get("token") or item.get("file_token") or "")


def safe_filename(value: str, fallback: str = "download.bin") -> str:
    name = Path(value).name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).rstrip(". ")
    return name or fallback


def inspect_pdf(path: Path) -> tuple[int, int]:
    try:
        reader = PdfReader(str(path))
        extracted_chars = sum(len(page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        raise FeishuProbeError(f"PDF 已下载，但解析失败：{exc}") from exc
    return len(reader.pages), extracted_chars


def run_probe(
    settings: FeishuProbeSettings,
    *,
    read_first_docx: bool = True,
    download_first_file: bool = False,
    download_dir: Path | None = None,
) -> None:
    client = FeishuReadOnlyClient(settings.app_id, settings.app_secret)
    client.authenticate()
    print("[成功] 应用凭证有效，已获取 tenant_access_token（令牌未显示）")

    for index, folder_token in enumerate(settings.folder_tokens, start=1):
        items = client.list_folder_items(folder_token)
        print(f"[成功] 文件夹 {index} 可访问，共发现 {len(items)} 个条目")
        for item in items:
            name = str(item.get("name") or "未命名")
            item_type = str(item.get("type") or "unknown")
            print(f"  - [{item_type}] {name}")

        if read_first_docx:
            docx_item = next(
                (item for item in items if item.get("type") == "docx"), None
            )
            if not docx_item:
                print("[提示] 当前层没有 docx 在线文档，跳过正文读取检查")
            else:
                document_id = _item_token(docx_item)
                if not document_id:
                    raise FeishuProbeError("找到 docx，但响应中没有文档 token")
                content = client.get_docx_raw_content(document_id)
                print(
                    f"[成功] 可读取在线文档《{docx_item.get('name') or '未命名'}》，"
                    f"正文长度 {len(content)} 字符（正文未显示）"
                )

        if not download_first_file:
            continue

        file_item = next((item for item in items if item.get("type") == "file"), None)
        if not file_item:
            print("[提示] 当前层没有普通文件，跳过下载检查")
            continue
        file_token = _item_token(file_item)
        if not file_token:
            raise FeishuProbeError("找到普通文件，但响应中没有文件 token")

        project_root = Path(__file__).resolve().parents[1]
        target_dir = download_dir or project_root / "data" / "runtime" / "feishu_probe"
        filename = safe_filename(str(file_item.get("name") or ""), f"{file_token}.bin")
        path, sha256, size = client.download_file(file_token, target_dir / filename)
        print(
            f"[成功] 已下载普通文件《{filename}》，大小 {size} 字节，"
            f"SHA-256 {sha256[:12]}…"
        )
        if path.suffix.lower() == ".pdf":
            pages, extracted_chars = inspect_pdf(path)
            print(
                f"[成功] PDF 可解析，共 {pages} 页，可提取文本 {extracted_chars} 字符"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只读检查飞书文件夹和在线文档访问权限")
    parser.add_argument(
        "--skip-docx",
        action="store_true",
        help="只列出文件，不读取第一篇 docx 正文",
    )
    parser.add_argument(
        "--download-first-file",
        action="store_true",
        help="下载每个配置文件夹里的第一个普通文件到 data/runtime/feishu_probe",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = load_settings()
        run_probe(
            settings,
            read_first_docx=not args.skip_docx,
            download_first_file=args.download_first_file,
        )
    except FeishuProbeError as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
