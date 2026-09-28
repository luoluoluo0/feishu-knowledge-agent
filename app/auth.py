from hmac import compare_digest

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import get_settings


# 前端或外部调用接口时，需要在请求头里带：
# X-API-Key: 你的接口 key
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


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
