from hmac import compare_digest

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings
from app.query_context import set_current_user
from app.users import SERVICE_USER_ID, SERVICE_USERNAME, AuthUser, decode_access_token


# 前端或外部调用接口时，需要在请求头里带：
# X-API-Key: 你的接口 key
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
# JWT 走标准的 Authorization: Bearer <token>。
bearer_header = HTTPBearer(auto_error=False)


def require_api_key(api_key: str | None = Security(api_key_header)) -> None:
    """校验普通接口 API Key。

    这是最小可用权限保护，不区分普通用户和管理员。
    后面如果要做管理员 key，可以在这里继续拆。
    """

    expected_key = get_settings().service_api_key
    if not expected_key:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_type": "auth_config_error",
                "message": "服务端没有配置 SERVICE_API_KEY。",
                "suggestion": "请在 .env 中配置 SERVICE_API_KEY 后重启服务。",
            },
        )

    if not api_key or not compare_digest(api_key, expected_key):
        raise HTTPException(
            status_code=401,
            detail={
                "success": False,
                "error_type": "unauthorized",
                "message": "API Key 不正确或没有传入。",
                "suggestion": "请在请求头 X-API-Key 中传入正确的接口 key。",
            },
        )


def _service_identity() -> AuthUser:
    """API Key 双轨回落的预留身份：旧脚本/评测不带 JWT 时落到 service
    用户（user_id=0）。它名下的旧 thread 对 JWT 用户一律 403，反向不受限。"""

    return AuthUser(
        user_id=SERVICE_USER_ID, username=SERVICE_USERNAME, via="api_key"
    )


def require_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_header),
    api_key: str | None = Security(api_key_header),
) -> AuthUser:
    """问答端点的用户依赖，双轨：

    - Authorization: Bearer <JWT> 有效 → 对应注册用户；
    - 无 JWT 但 X-API-Key 正确 → service 遗留身份（旧脚本不破）；
    - 都没有/都无效 → 401。

    解析成功即把用户写入 ContextVar，本次请求内的工具调用（同线程）
    可读取；跨线程路径（SSE worker）须在 worker 首行重设。
    """

    expected_key = get_settings().service_api_key
    if credentials and credentials.credentials:
        user = decode_access_token(credentials.credentials)
        if user is not None:
            set_current_user(user.user_id)
            return user

    if (
        expected_key
        and api_key
        and compare_digest(api_key, expected_key)
    ):
        service = _service_identity()
        set_current_user(service.user_id)
        return service

    raise HTTPException(
        status_code=401,
        detail={
            "success": False,
            "error_type": "unauthorized",
            "message": "未登录或凭证已失效。",
            "suggestion": "请先登录获取令牌，请求头带 Authorization: Bearer <token>；"
            "或以正确的 X-API-Key 走管理通道。",
        },
    )
