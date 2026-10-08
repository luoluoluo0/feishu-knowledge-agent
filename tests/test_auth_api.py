"""认证端点与数据隔离的 API 层测试。

require_current_user 走真实双轨逻辑（JWT / API Key），只 override 限流；
users 库通过 monkeypatch get_settings 指到 tmp_path——CI 无 .env 必须原样通过。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import api as api_module
from app.rate_limit import require_rate_limit


@pytest.fixture
def client(tmp_path, monkeypatch, make_settings):
    settings = make_settings(
        conversation_db_path=str(tmp_path / "history.db"),
        jwt_secret="test-jwt-secret",
        service_api_key="test-service-key",
    )
    monkeypatch.setattr("app.users.get_settings", lambda: settings)
    monkeypatch.setattr("app.auth.get_settings", lambda: settings)
    # Token 用量读 agent_logs.db（真实路径是项目 data/runtime）：
    # 指到 tmp_path，测试不碰真库，也不依赖它有没有跑过迁移。
    log_db = tmp_path / "agent_logs.db"
    monkeypatch.setattr("app.logging_store.LOG_DB_PATH", log_db)
    monkeypatch.setattr(
        api_module,
        "clear_thread_checkpoints",
        lambda thread_id: {
            "thread_id": thread_id,
            "db_exists": False,
            "deleted_checkpoints": 0,
            "deleted_writes": 0,
        },
    )
    monkeypatch.setattr(api_module, "clear_history", lambda thread_id: 3)
    api_module.app.dependency_overrides[require_rate_limit] = lambda: None
    yield TestClient(api_module.app)
    api_module.app.dependency_overrides.clear()


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_register_login_me_roundtrip(client):
    register = client.post(
        "/auth/register", json={"username": "alice", "password": "password1"}
    )
    assert register.status_code == 200
    token = register.json()["token"]
    assert register.json()["user"]["username"] == "alice"

    me = client.get("/auth/me", headers=_auth_headers(token))
    assert me.status_code == 200
    assert me.json()["user"]["username"] == "alice"
    assert me.json()["via"] == "jwt"

    login = client.post(
        "/auth/login", json={"username": "alice", "password": "password1"}
    )
    assert login.status_code == 200
    assert login.json()["token"]


def test_register_duplicate_returns_409(client):
    body = {"username": "bob", "password": "password1"}
    assert client.post("/auth/register", json=body).status_code == 200

    again = client.post("/auth/register", json=body)

    assert again.status_code == 409


def test_login_wrong_password_returns_401(client):
    client.post("/auth/register", json={"username": "carol", "password": "password1"})

    bad = client.post(
        "/auth/login", json={"username": "carol", "password": "wrong-pass"}
    )

    assert bad.status_code == 401
    assert bad.json()["detail"]["error_type"] == "invalid_credentials"


def test_me_without_credentials_returns_401(client):
    assert client.get("/auth/me").status_code == 401
    assert (
        client.get(
            "/auth/me", headers=_auth_headers("not-a-real-token")
        ).status_code
        == 401
    )


def test_api_key_dual_track_resolves_service_identity(client):
    """双轨：老脚本只带 X-API-Key 也能用问答端点，身份落到 service。"""

    me = client.get("/auth/me", headers={"X-API-Key": "test-service-key"})

    assert me.status_code == 200
    assert me.json()["via"] == "api_key"
    assert me.json()["user"]["username"] == "service"


def test_history_isolated_between_users(client):
    """数据隔离：A 先用某 thread，B 再访问必须 403，A 自己永远可以。"""

    token_a = client.post(
        "/auth/register", json={"username": "alice", "password": "password1"}
    ).json()["token"]
    token_b = client.post(
        "/auth/register", json={"username": "bob", "password": "password1"}
    ).json()["token"]

    first = client.get(
        "/auth/history/thread-shared", headers=_auth_headers(token_a)
    )
    assert first.status_code == 200

    forbidden = client.get(
        "/auth/history/thread-shared", headers=_auth_headers(token_b)
    )
    assert forbidden.status_code == 403
    assert forbidden.json()["detail"]["error_type"] == "thread_forbidden"

    again = client.get(
        "/auth/history/thread-shared", headers=_auth_headers(token_a)
    )
    assert again.status_code == 200


def test_sessions_list_only_own_threads(client):
    token_a = client.post(
        "/auth/register", json={"username": "alice", "password": "password1"}
    ).json()["token"]
    token_b = client.post(
        "/auth/register", json={"username": "bob", "password": "password1"}
    ).json()["token"]
    client.get("/auth/history/thread-a-1", headers=_auth_headers(token_a))
    client.get("/auth/history/thread-a-2", headers=_auth_headers(token_a))
    client.get("/auth/history/thread-b-1", headers=_auth_headers(token_b))

    sessions_a = client.get("/auth/sessions", headers=_auth_headers(token_a)).json()
    sessions_b = client.get("/auth/sessions", headers=_auth_headers(token_b)).json()

    assert {s["thread_id"] for s in sessions_a["sessions"]} == {"thread-a-1", "thread-a-2"}
    assert {s["thread_id"] for s in sessions_b["sessions"]} == {"thread-b-1"}


def test_clear_thread_owner_check(client):
    """JWT 用户只能清自己的会话；service 身份（API Key）不受限。"""

    token_a = client.post(
        "/auth/register", json={"username": "alice", "password": "password1"}
    ).json()["token"]
    token_b = client.post(
        "/auth/register", json={"username": "bob", "password": "password1"}
    ).json()["token"]
    client.get("/auth/history/thread-alice", headers=_auth_headers(token_a))

    forbidden = client.post(
        "/admin/clear-thread",
        json={"thread_id": "thread-alice"},
        headers=_auth_headers(token_b),
    )
    assert forbidden.status_code == 403

    allowed = client.post(
        "/admin/clear-thread",
        json={"thread_id": "thread-alice"},
        headers=_auth_headers(token_a),
    )
    assert allowed.status_code == 200
    assert allowed.json()["success"] is True

    admin = client.post(
        "/admin/clear-thread",
        json={"thread_id": "thread-alice"},
        headers={"X-API-Key": "test-service-key"},
    )
    assert admin.status_code == 200


def test_stats_shape_and_token_attribution(client, tmp_path):
    token = client.post(
        "/auth/register", json={"username": "stats-user", "password": "password1"}
    ).json()["token"]

    # 直接往（已隔离的）日志库插一条该用户的请求：stats 的 Token 归因
    # 必须认它——这正是工作台「Token 使用情况」卡的数据来源。
    import sqlite3
    from app.logging_store import LOG_DB_PATH, init_log_db

    init_log_db()
    connection = sqlite3.connect(LOG_DB_PATH)
    connection.execute(
        "INSERT INTO agent_logs (created_at, mode, thread_id, question, answer, "
        "trace_json, success, error, latency_ms, total_tokens, user_id) "
        "VALUES ('2026-09-29T10:00:00', 'tool_agent', 't-x', 'q', 'a', '{}', 1, '', 5, 123, ?)",
        (_current_stats_user_id(client, token),),
    )
    connection.commit()
    connection.close()

    stats = client.get("/auth/stats", headers=_auth_headers(token)).json()

    assert stats["success"] is True
    assert stats["tokens"]["total_tokens"] == 123
    assert stats["tokens"]["requests"] == 1
    assert stats["sessions"] == 0
    assert "library" in stats


def _current_stats_user_id(client, token):
    me = client.get("/auth/me", headers=_auth_headers(token)).json()
    return me["user"]["id"]


def test_feedback_rating_validation_and_store(client, tmp_path):
    token = client.post(
        "/auth/register", json={"username": "fb-user", "password": "password1"}
    ).json()["token"]
    headers = _auth_headers(token)

    bad = client.post(
        "/auth/feedback",
        json={"thread_id": "t", "question": "q", "rating": 5},
        headers=headers,
    )
    assert bad.status_code == 400

    good = client.post(
        "/auth/feedback",
        json={"thread_id": "t-1", "question": "这个结论可靠吗", "rating": 1},
        headers=headers,
    )
    assert good.status_code == 200

    from app.config import get_settings as real_get_settings
    import sqlite3

    db = tmp_path / "users.db"
    assert db.exists()
    connection = sqlite3.connect(db)
    row = connection.execute(
        "SELECT rating, question FROM feedback"
    ).fetchone()
    connection.close()
    assert row[0] == 1 and "结论" in row[1]


def test_library_endpoint_runs_without_index(client):
    """CI 无文献索引 CSV：端点必须空列表正常返回，不能炸。"""

    token = client.post(
        "/auth/register", json={"username": "lib-user", "password": "password1"}
    ).json()["token"]

    response = client.get("/auth/library", headers=_auth_headers(token))
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert isinstance(body["items"], list)
    # 本地有索引 CSV 时 items 非空、CI 上为空——两种环境都只验结构
    for item in body["items"]:
        assert item["item_id"] and "favorite" in item


def test_favorites_roundtrip_and_isolation(client):
    token_a = client.post(
        "/auth/register", json={"username": "fav-a", "password": "password1"}
    ).json()["token"]
    token_b = client.post(
        "/auth/register", json={"username": "fav-b", "password": "password1"}
    ).json()["token"]
    headers_a = _auth_headers(token_a)
    headers_b = _auth_headers(token_b)

    add = client.post("/auth/favorites", json={"item_id": "12"}, headers=headers_a)
    assert add.status_code == 200
    assert add.json()["item_id"] == "012"  # 归一化为三位编号

    favorites_a = client.get("/auth/favorites", headers=headers_a).json()
    assert [f["item_id"] for f in favorites_a["favorites"]] == ["012"]

    favorites_b = client.get("/auth/favorites", headers=headers_b).json()
    assert favorites_b["favorites"] == []

    remove = client.delete("/auth/favorites/012", headers=headers_a)
    assert remove.status_code == 200
    assert remove.json()["favorite"] is False

    favorites_after = client.get("/auth/favorites", headers=headers_a).json()
    assert favorites_after["favorites"] == []
