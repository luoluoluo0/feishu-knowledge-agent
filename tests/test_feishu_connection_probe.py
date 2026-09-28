from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_feishu_connection import (
    FeishuProbeError,
    FeishuProbeSettings,
    FeishuReadOnlyClient,
    load_settings,
    safe_filename,
    split_folder_tokens,
)


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self._payload

    def iter_content(self, chunk_size):
        del chunk_size
        yield b"first-"
        yield b"chunk"


class FakeSession:
    def __init__(self):
        self.post_calls = []
        self.get_calls = []

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return FakeResponse({"code": 0, "tenant_access_token": "tenant-token"})

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        if url.endswith("/drive/v1/files"):
            return FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "files": [
                            {"name": "测试在线文档", "type": "docx", "token": "doc-token"}
                        ],
                        "has_more": False,
                    },
                }
            )
        return FakeResponse({"code": 0, "data": {"content": "测试正文"}})


def test_split_folder_tokens_accepts_common_separators():
    assert split_folder_tokens("folder-a, folder-b;folder-c\nfolder-d") == (
        "folder-a",
        "folder-b",
        "folder-c",
        "folder-d",
    )


def test_load_settings_reports_names_but_not_secret(tmp_path, monkeypatch):
    for key in ("FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_FOLDER_TOKENS"):
        monkeypatch.delenv(key, raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("FEISHU_APP_ID=app-id\nFEISHU_APP_SECRET=very-secret\n", encoding="utf-8")

    with pytest.raises(FeishuProbeError) as exc_info:
        load_settings(env_path)

    message = str(exc_info.value)
    assert "FEISHU_FOLDER_TOKENS" in message
    assert "very-secret" not in message


def test_client_authenticates_lists_files_and_reads_docx():
    session = FakeSession()
    client = FeishuReadOnlyClient("app-id", "app-secret", session=session)

    client.authenticate()
    items = client.list_folder_items("folder-token")
    content = client.get_docx_raw_content("doc-token")

    assert items == [{"name": "测试在线文档", "type": "docx", "token": "doc-token"}]
    assert content == "测试正文"
    assert session.get_calls[0][1]["headers"] == {
        "Authorization": "Bearer tenant-token"
    }


def test_client_downloads_file_atomically_and_hashes_it(tmp_path):
    session = FakeSession()
    client = FeishuReadOnlyClient("app-id", "app-secret", session=session)
    client.authenticate()

    path, sha256, size = client.download_file("file-token", tmp_path / "paper.pdf")

    assert path.read_bytes() == b"first-chunk"
    assert sha256 == "9998dc341cff6e4a6c77f7457572670845417d1e0c8cbeeb6ce92be94ed90dfc"
    assert size == 11
    assert not (tmp_path / "paper.pdf.part").exists()


def test_safe_filename_blocks_directory_traversal_and_windows_characters():
    assert safe_filename("../../bad:name?.pdf") == "bad_name_.pdf"


def test_api_error_does_not_expose_credentials():
    class ErrorSession(FakeSession):
        def get(self, url, **kwargs):
            return FakeResponse(
                {"code": 99991672, "msg": "Access denied"},
                status_code=403,
                headers={"X-Tt-Logid": "log-id"},
            )

    client = FeishuReadOnlyClient("app-id", "do-not-leak", session=ErrorSession())
    client.authenticate()

    with pytest.raises(FeishuProbeError) as exc_info:
        client.list_folder_items("folder-token")

    message = str(exc_info.value)
    assert "Access denied" in message
    assert "log-id" in message
    assert "do-not-leak" not in message
    assert "tenant-token" not in message


def test_settings_dataclass_keeps_multiple_folders():
    settings = FeishuProbeSettings("app", "secret", ("folder-a", "folder-b"))
    assert settings.folder_tokens == ("folder-a", "folder-b")
