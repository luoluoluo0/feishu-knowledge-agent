from dataclasses import asdict, dataclass

import requests


@dataclass
class ClassifiedError:
    """接口层返回给前端的错误信息。"""

    error_type: str
    message: str
    technical_detail: str
    suggestion: str
    status_code: int

    def to_detail(self) -> dict:
        """转成 FastAPI HTTPException 可以返回的 JSON。"""

        data = asdict(self)
        data.pop("status_code", None)
        data["success"] = False
        return data


def _contains(text: str, *keywords: str) -> bool:
    """判断文本里是否包含任意关键词，统一转小写比较。"""

    lower_text = text.lower()
    return any(keyword.lower() in lower_text for keyword in keywords)


def classify_exception(exc: Exception) -> ClassifiedError:
    """把杂乱的底层异常分类成用户能看懂的接口错误。

    这里不追求覆盖所有异常，只先覆盖项目里最常见的几类：
    配置错误、模型认证错误、模型限流、Qdrant 检索错误、超时错误。
    """

    error_name = exc.__class__.__name__
    technical_detail = str(exc)
    combined = f"{error_name}: {technical_detail}"

    if isinstance(exc, ValueError) and _contains(combined, ".env", "api_key", "key"):
        return ClassifiedError(
            error_type="config_error",
            message="配置读取失败，请检查 .env 里的 API Key 和服务配置。",
            technical_detail=technical_detail,
            suggestion="确认 DEEPSEEK_API_KEY、SILICON_API_KEY、QDRANT_URL 是否存在且没有多余引号。",
            status_code=400,
        )

    if _contains(combined, "authentication", "invalid api key", "api key", "error code: 401"):
        return ClassifiedError(
            error_type="model_auth_error",
            message="模型接口认证失败，API Key 可能无效或没有正确传入。",
            technical_detail=technical_detail,
            suggestion="检查 .env 中的 DEEPSEEK_API_KEY / SILICON_API_KEY，并重启 FastAPI 服务。",
            status_code=401,
        )

    if _contains(combined, "permission", "access denied", "error code: 403"):
        return ClassifiedError(
            error_type="model_permission_error",
            message="模型接口没有权限访问当前服务或模型。",
            technical_detail=technical_detail,
            suggestion="检查账号权限、实名状态、模型名称和 base_url 是否匹配。",
            status_code=403,
        )

    if _contains(combined, "rate limit", "too many requests", "error code: 429"):
        return ClassifiedError(
            error_type="model_rate_limit",
            message="模型接口请求过快或额度受限。",
            technical_detail=technical_detail,
            suggestion="稍等一会儿再试，或者降低并发请求次数。",
            status_code=429,
        )

    if isinstance(exc, requests.exceptions.Timeout) or _contains(combined, "timeout", "timed out"):
        return ClassifiedError(
            error_type="timeout_error",
            message="请求超时，可能是模型接口或 Qdrant 响应太慢。",
            technical_detail=technical_detail,
            suggestion="检查网络、Qdrant 服务状态，或者稍后重试。",
            status_code=504,
        )

    if isinstance(exc, requests.exceptions.ConnectionError) or _contains(
        combined,
        "qdrant",
        "6333",
        "points/search",
        "connection refused",
        "failed to establish",
    ):
        return ClassifiedError(
            error_type="retrieval_error",
            message="向量检索服务连接失败或检索异常。",
            technical_detail=technical_detail,
            suggestion="确认 Qdrant 容器已经启动，并且 QDRANT_URL 指向 http://127.0.0.1:6333。",
            status_code=503,
        )

    if _contains(combined, "embedding", "silicon", "20015"):
        return ClassifiedError(
            error_type="embedding_error",
            message="问题向量化失败，embedding 接口返回了错误。",
            technical_detail=technical_detail,
            suggestion=(
                "该接口会偶发返回 code 20015，同一段文本重试通常就能成功，"
                "调用处已自动重试 3 次。若仍持续失败，再检查 EMBEDDINGS_MODEL、"
                "EMBEDDINGS_BASE_URL 与账号额度。"
            ),
            status_code=502,
        )

    # 裸 "400" 不能当特征：端口号、行号、大小里到处是它。
    # 只认 SDK 实际会打出的形态。
    if _contains(combined, "badrequest", "bad request", "error code: 400"):
        return ClassifiedError(
            error_type="model_request_error",
            message="模型接口请求参数不符合要求。",
            technical_detail=technical_detail,
            suggestion="检查模型名称、base_url、response_format 或工具调用参数。",
            status_code=400,
        )

    return ClassifiedError(
        error_type="internal_error",
        message="服务内部错误，暂时无法完成这次请求。",
        technical_detail=technical_detail,
        suggestion="查看日志里的 technical_detail，定位具体是哪一层抛出的异常。",
        status_code=500,
    )
