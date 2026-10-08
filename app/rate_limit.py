import hashlib
import threading
import time

from fastapi import HTTPException, Request

from app.config import get_settings


# 进程内固定窗口限流。
# 适合本地学习和单进程服务；如果以后多机器部署，再换集中式方案。
#
# 之前这里用的是 Redis：但地址写死 127.0.0.1:6379、requirements 里没声明
# redis 包（换机直接 ImportError）、Redis 挂了所有受保护接口跟着 500，
# 还把调用方的 API Key 明文拼进键名并 print 到 stdout。对单进程服务来说
# 这些代价换不来任何好处，退回内存版。


_WINDOW_SECONDS = 60

_lock = threading.Lock()
# key -> (窗口内已计数, 窗口起点)
_counters: dict[str, tuple[int, float]] = {}


def _client_key(request: Request) -> str:
    """用 IP + 凭证组成限流身份。

    凭证取「Bearer JWT 优先，X-API-Key 兜底」——与认证双轨同序。JWT 每
    用户一枚，登录用户天然各占一个桶，互不挤占额度；旧脚本共用 API Key
    则共享一个桶（与单 key 时代行为一致）。完整凭证只用来算哈希，不进
    任何字符串——键名、日志、报错里都不能出现原文。
    """

    client_host = request.client.host if request.client else "unknown"
    authorization = request.headers.get("Authorization", "")
    if authorization.startswith("Bearer "):
        credential = authorization[7:].strip()
    else:
        credential = request.headers.get("X-API-Key", "")
    credential_digest = hashlib.sha256(credential.encode("utf-8")).hexdigest()[:8]
    return f"{client_host}:{credential_digest}"


def _prune_expired(now: float) -> None:
    """清掉已过期的窗口，防止长期运行时字典无限膨胀。"""

    expired = [
        key
        for key, (_, window_start) in _counters.items()
        if now - window_start >= _WINDOW_SECONDS
    ]
    for key in expired:
        _counters.pop(key, None)


def require_rate_limit(request: Request) -> None:
    """限制同一个调用方每分钟的请求次数。"""

    settings = get_settings()
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return

    key = _client_key(request)
    now = time.monotonic()

    with _lock:
        _prune_expired(now)

        count, window_start = _counters.get(key, (0, now))
        if now - window_start >= _WINDOW_SECONDS:
            count, window_start = 0, now

        count += 1
        _counters[key] = (count, window_start)
        retry_after = max(1, int(_WINDOW_SECONDS - (now - window_start)))

    if count > limit:
        raise HTTPException(
            status_code=429,
            detail={
                "success": False,
                "error_type": "rate_limit",
                "message": f"请求太频繁，每分钟最多允许 {limit} 次。",
                "suggestion": (
                    "稍等一会儿再发送，或者在 .env 中调整 "
                    f"FEISHU_RATE_LIMIT_PER_MINUTE。请等待 {retry_after} 秒后再尝试。"
                ),
            },
        )
