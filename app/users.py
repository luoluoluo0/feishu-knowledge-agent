"""用户账号与 thread 归属存储。

设计对齐 conversation_store：函数显式接收 settings、contextmanager 开库、
表结构 IF NOT EXISTS、异常不向上抛（注册/登录返回结果对象而不是 raise，
让 API 层决定映射成哪个状态码）。

两张表：
- users：账号本体。密码存 pbkdf2 哈希（salt$hash，hex），绝不落明文。
- sessions：thread_id 归属登记。隔离模型=「thread 首次被某用户使用即注册
  归属，他人再访问判 403」；conversation_turns 不动 schema，隔离在
  thread 边界执行。
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import jwt

from app.config import Settings, get_settings

_PBKDF2_ITERATIONS = 200_000

# API Key 双轨回落的预留身份：旧脚本/评测不带 JWT 时落到这个用户，
# 它名下的 thread 对 JWT 用户一律 403，反向（service 身份）不受限——
# 管理通道本就能清任何 thread。
SERVICE_USER_ID = 0
SERVICE_USERNAME = "service"


class UserError(Exception):
    """注册/登录失败，message 面向客户端。"""

    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class ThreadAccessError(Exception):
    """当前用户不是 thread 的归属者。"""

    def __init__(self, thread_id: str):
        super().__init__(f"thread {thread_id} 不属于当前用户")
        self.thread_id = thread_id


@dataclass
class AuthUser:
    """依赖注入到端点的最小身份。"""

    user_id: int
    username: str
    via: str  # "jwt" | "api_key"


def get_users_db_path(settings: Settings | None = None) -> Path:
    """用户库与对话库同目录（data/runtime/users.db），随设置走。"""

    settings = settings or get_settings()
    return Path(settings.conversation_db_path).parent / "users.db"


@contextmanager
def open_db(settings: Settings | None = None):
    db_path = get_users_db_path(settings)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS user_sessions (
                thread_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                last_active_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_user_sessions_user
                ON user_sessions (user_id, thread_id);
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                thread_id TEXT NOT NULL DEFAULT '',
                question TEXT NOT NULL DEFAULT '',
                rating INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS favorites (
                user_id INTEGER NOT NULL,
                item_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (user_id, item_id)
            );
            """
        )
        connection.commit()
        yield connection
    finally:
        connection.close()


# ---------- 密码哈希（pbkdf2，标准库，无额外依赖） ----------


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    ).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, digest = stored.split("$", 1)
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    ).hex()
    return hmac.compare_digest(candidate, digest)


# ---------- 账号 ----------


def register_user(username: str, password: str, settings=None) -> AuthUser:
    """注册新用户。用户名重复抛 UserError(conflict)。"""

    username = (username or "").strip()
    if not username or not password:
        raise UserError("invalid_input", "用户名和密码不能为空。", 400)
    if len(password) < 6:
        raise UserError("weak_password", "密码至少 6 位。", 400)
    with open_db(settings) as connection:
        exists = connection.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        ).fetchone()
        if exists:
            raise UserError("username_taken", "用户名已被注册。", 409)
        cursor = connection.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (
                username,
                hash_password(password),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        connection.commit()
        return AuthUser(user_id=int(cursor.lastrowid), username=username, via="register")


def authenticate_user(username: str, password: str, settings=None) -> AuthUser:
    """校验用户名密码。失败统一抛 UserError(invalid_credentials)，不区分
    「用户不存在」和「密码错误」，避免用户名枚举。"""

    with open_db(settings) as connection:
        row = connection.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            ((username or "").strip(),),
        ).fetchone()
    if row is None or not verify_password(password or "", row["password_hash"]):
        raise UserError("invalid_credentials", "用户名或密码不正确。", 401)
    return AuthUser(user_id=int(row["id"]), username=row["username"], via="login")


# ---------- JWT ----------


def _jwt_secret(settings: Settings | None = None) -> str:
    """签名密钥。未配置 JWT_SECRET 时临时随机生成并缓存到模块级——
    语义是「重启后所有登录失效」，比用固定默认值安全（伪造不了）。"""

    global _ephemeral_secret
    settings = settings or get_settings()
    if settings.jwt_secret:
        return settings.jwt_secret
    if not _ephemeral_secret:
        import logging

        logging.getLogger(__name__).warning(
            "JWT_SECRET 未配置，使用进程内临时密钥：重启后所有已登录用户需重新登录。"
        )
        _ephemeral_secret = secrets.token_hex(32)
    return _ephemeral_secret


_ephemeral_secret = ""


def create_access_token(user_id: int, username: str, settings=None) -> str:
    settings = settings or get_settings()
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + settings.jwt_expire_minutes * 60,
    }
    return jwt.encode(payload, _jwt_secret(settings), algorithm="HS256")


def decode_access_token(token: str, settings=None) -> AuthUser | None:
    """解码并校验 JWT。过期/签名错/格式错一律返回 None，由调用方决定 401。"""

    try:
        payload = jwt.decode(
            token, _jwt_secret(settings), algorithms=["HS256"]
        )
    except jwt.PyJWTError:
        return None
    user_id = payload.get("sub")
    username = payload.get("username") or ""
    if not user_id or not str(user_id).isdigit():
        return None
    return AuthUser(user_id=int(user_id), username=username, via="jwt")


# ---------- thread 归属（数据隔离的边界） ----------


def register_thread_owner(thread_id: str, user_id: int, settings=None) -> bool:
    """登记 thread 归属。返回 True=本次新注册（首次使用），False=已归属
    任何人（调用方必须再查 get_thread_owner 判断是不是自己）。"""

    thread_id = (thread_id or "").strip()
    if not thread_id:
        return False
    now = datetime.now(timezone.utc).isoformat()
    with open_db(settings) as connection:
        cursor = connection.execute(
            "INSERT OR IGNORE INTO user_sessions "
            "(thread_id, user_id, created_at, last_active_at) VALUES (?, ?, ?, ?)",
            (thread_id, user_id, now, now),
        )
        registered = cursor.rowcount == 1
        if not registered:
            connection.execute(
                "UPDATE user_sessions SET last_active_at = ? WHERE thread_id = ?",
                (now, thread_id),
            )
        connection.commit()
    return registered


def get_thread_owner(thread_id: str, settings=None) -> int | None:
    with open_db(settings) as connection:
        row = connection.execute(
            "SELECT user_id FROM user_sessions WHERE thread_id = ?",
            ((thread_id or "").strip(),),
        ).fetchone()
    return int(row["user_id"]) if row else None


def ensure_thread_owner(thread_id: str, user_id: int, settings=None) -> None:
    """归属校验：未登记则登记为当前用户；已登记且不是当前用户抛
    ThreadAccessError（API 层映射 403，水平越权防护点）。service 身份
    （user_id=0）不受限——管理通道本就能操作任何 thread。"""

    if user_id == SERVICE_USER_ID:
        return
    if not register_thread_owner(thread_id, user_id, settings):
        owner = get_thread_owner(thread_id, settings)
        if owner is not None and owner != user_id:
            raise ThreadAccessError(thread_id)


def list_user_threads(user_id: int, limit: int = 100, settings=None) -> list[dict]:
    with open_db(settings) as connection:
        rows = connection.execute(
            "SELECT thread_id, created_at, last_active_at FROM user_sessions "
            "WHERE user_id = ? ORDER BY last_active_at DESC LIMIT ?",
            (user_id, max(1, min(limit, 500))),
        ).fetchall()
    return [dict(row) for row in rows]


# ---------- 业务扩展：回答反馈与文献收藏 ----------


def add_feedback(
    user_id: int,
    thread_id: str,
    question: str,
    rating: int,
    settings=None,
) -> None:
    """记录用户对某轮回答的 👍/👎。评分合法性由 API 层校验。"""

    with open_db(settings) as connection:
        connection.execute(
            "INSERT INTO feedback (user_id, thread_id, question, rating, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                user_id,
                (thread_id or "").strip(),
                (question or "").strip()[:500],
                rating,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        connection.commit()


def add_favorite(user_id: int, item_id: str, settings=None) -> None:
    with open_db(settings) as connection:
        connection.execute(
            "INSERT OR IGNORE INTO favorites (user_id, item_id, created_at) "
            "VALUES (?, ?, ?)",
            (user_id, item_id, datetime.now(timezone.utc).isoformat()),
        )
        connection.commit()


def remove_favorite(user_id: int, item_id: str, settings=None) -> bool:
    with open_db(settings) as connection:
        cursor = connection.execute(
            "DELETE FROM favorites WHERE user_id = ? AND item_id = ?",
            (user_id, item_id),
        )
        connection.commit()
    return cursor.rowcount > 0


def list_favorites(user_id: int, settings=None) -> list[dict]:
    with open_db(settings) as connection:
        rows = connection.execute(
            "SELECT item_id, created_at FROM favorites WHERE user_id = ? "
            "ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def user_stats(user_id: int, settings=None) -> dict:
    """个人工作台统计：会话数与累计提问数。

    会话在 users.db、台账在 conversation_history.db——跨库两步查询：
    先取本人 thread 列表，再去台账库数行数。数量级小，不值得 ATTACH。
    """

    settings = settings or get_settings()
    with open_db(settings) as connection:
        thread_rows = connection.execute(
            "SELECT thread_id FROM user_sessions WHERE user_id = ?", (user_id,)
        ).fetchall()
    thread_ids = [row["thread_id"] for row in thread_rows]

    questions = 0
    if thread_ids:
        conversation_path = Path(settings.conversation_db_path)
        if conversation_path.exists():
            conversation = sqlite3.connect(conversation_path)
            try:
                placeholders = ",".join("?" for _ in thread_ids)
                row = conversation.execute(
                    "SELECT COUNT(*) FROM conversation_turns "
                    f"WHERE thread_id IN ({placeholders})",
                    thread_ids,
                ).fetchone()
                questions = int(row[0]) if row else 0
            finally:
                conversation.close()

    return {"sessions": len(thread_ids), "questions": questions}


def get_token_usage(user_id: int, settings=None) -> dict:
    """个人 Token 用量：从 agent_logs 按用户聚合（累计 + 今日）。

    agent_logs.db 的路径是 logging_store 的固定常量；service 遗留身份
    （user_id=0）聚合的是历史上无主请求，工作台照常展示。
    """

    from app.logging_store import LOG_DB_PATH, init_log_db

    settings = settings or get_settings()
    del settings  # 路径当前固定，保留参数与其它函数签名一致
    result = {
        "total_tokens": 0,
        "today_tokens": 0,
        "requests": 0,
    }
    if not LOG_DB_PATH.exists():
        return result
    # 老库文件可能还没有 user_id 列：先跑一次幂等迁移再查。
    init_log_db()

    today = datetime.now(timezone.utc).date().isoformat()
    connection = sqlite3.connect(LOG_DB_PATH)
    try:
        row = connection.execute(
            "SELECT COALESCE(SUM(total_tokens), 0), COUNT(*) FROM agent_logs "
            "WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        result["total_tokens"] = int(row[0] or 0)
        result["requests"] = int(row[1] or 0)
        row = connection.execute(
            "SELECT COALESCE(SUM(total_tokens), 0) FROM agent_logs "
            "WHERE user_id = ? AND substr(created_at, 1, 10) = ?",
            (user_id, today),
        ).fetchone()
        result["today_tokens"] = int(row[0] or 0)
    finally:
        connection.close()
    return result
