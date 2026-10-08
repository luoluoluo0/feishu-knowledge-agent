"""用户账号、JWT 与 thread 归属的单元测试。

全部走 conftest 的 make_settings（不读 .env）+ tmp_path 数据库，
CI 裸跑环境必须原样通过（发布铁律）。
"""

from __future__ import annotations

import dataclasses
import time

import pytest

from app.users import (
    SERVICE_USER_ID,
    AuthUser,
    ThreadAccessError,
    UserError,
    create_access_token,
    decode_access_token,
    ensure_thread_owner,
    get_thread_owner,
    list_user_threads,
    register_thread_owner,
    register_user,
    hash_password,
    verify_password,
    authenticate_user,
)


@pytest.fixture
def settings(tmp_path, make_settings):
    """用户库指到 tmp_path，JWT 密钥显式配置（确定性）。"""

    return make_settings(
        conversation_db_path=str(tmp_path / "users" / "history.db"),
        jwt_secret="test-jwt-secret",
        jwt_expire_minutes=60,
    )


# ---------- 密码哈希 ----------


def test_password_hash_roundtrip():
    stored = hash_password("s3cret-密码")

    assert stored != "s3cret-密码"
    assert verify_password("s3cret-密码", stored) is True
    assert verify_password("wrong", stored) is False


def test_password_hash_unique_salt_per_call():
    assert hash_password("same") != hash_password("same")


# ---------- 注册与登录 ----------


def test_register_and_authenticate_roundtrip(settings):
    user = register_user("alice", "password1", settings)

    assert user.user_id > 0
    assert user.username == "alice"

    login = authenticate_user("alice", "password1", settings)
    assert login.user_id == user.user_id


def test_register_rejects_duplicate_username(settings):
    register_user("bob", "password1", settings)

    with pytest.raises(UserError) as excinfo:
        register_user("bob", "other-pass", settings)

    assert excinfo.value.status_code == 409


def test_register_rejects_weak_password(settings):
    with pytest.raises(UserError) as excinfo:
        register_user("carol", "123", settings)

    assert excinfo.value.status_code == 400


def test_authenticate_failure_is_indistinguishable(settings):
    """用户不存在和密码错误必须返回同一个错误——防用户名枚举。"""

    register_user("dave", "password1", settings)

    for kwargs in [
        {"username": "ghost", "password": "whatever"},
        {"username": "dave", "password": "wrong-pass"},
    ]:
        with pytest.raises(UserError) as excinfo:
            authenticate_user(settings=settings, **kwargs)
        assert excinfo.value.status_code == 401
        assert excinfo.value.code == "invalid_credentials"


# ---------- JWT ----------


def test_token_roundtrip(settings):
    token = create_access_token(7, "alice", settings)

    user = decode_access_token(token, settings)

    assert user is not None
    assert user.user_id == 7
    assert user.username == "alice"
    assert user.via == "jwt"


def test_tampered_token_rejected(settings):
    token = create_access_token(7, "alice", settings)

    assert decode_access_token(token + "x", settings) is None


def test_expired_token_rejected(settings):
    expired_settings = dataclasses.replace(settings, jwt_expire_minutes=-1)
    token = create_access_token(7, "alice", expired_settings)
    time.sleep(1.1)  # exp 是整秒，留出过期的余量

    assert decode_access_token(token, settings) is None


def test_token_bound_to_secret(settings):
    token = create_access_token(7, "alice", settings)
    other = dataclasses.replace(settings, jwt_secret="another-secret")

    assert decode_access_token(token, other) is None


# ---------- thread 归属（数据隔离边界） ----------


def test_first_use_registers_ownership(settings):
    assert register_thread_owner("thread-a", 1, settings) is True
    assert get_thread_owner("thread-a", settings) == 1


def test_second_user_access_raises(settings):
    register_thread_owner("thread-a", 1, settings)

    with pytest.raises(ThreadAccessError):
        ensure_thread_owner("thread-a", 2, settings)


def test_owner_reaccess_is_fine(settings):
    register_thread_owner("thread-a", 1, settings)

    ensure_thread_owner("thread-a", 1, settings)  # 不抛即通过


def test_service_identity_bypasses_ownership(settings):
    """service（API Key 遗留身份）是管理通道，不受归属限制。"""

    register_thread_owner("thread-a", 1, settings)

    ensure_thread_owner("thread-a", SERVICE_USER_ID, settings)


def test_service_owned_threads_are_off_limits_to_users(settings):
    """旧脚本经 service 身份创建的 thread，JWT 用户不可见。"""

    register_thread_owner("legacy-thread", SERVICE_USER_ID, settings)

    with pytest.raises(ThreadAccessError):
        ensure_thread_owner("legacy-thread", 1, settings)


def test_list_user_threads_only_lists_own(settings):
    register_thread_owner("t-1", 1, settings)
    register_thread_owner("t-2", 1, settings)
    register_thread_owner("t-3", 2, settings)

    threads = list_user_threads(1, settings=settings)

    assert {t["thread_id"] for t in threads} == {"t-1", "t-2"}


def test_empty_thread_id_is_ignored(settings):
    assert register_thread_owner("", 1, settings) is False
    assert get_thread_owner("", settings) is None
