import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from app.config import Settings, get_settings
from app.token_meter import TokenMeter


logger = logging.getLogger(__name__)

_langfuse_client: Langfuse | None = None
_langfuse_config_key: tuple[str, str, str, str] | None = None


def validate_langfuse_settings(settings: Settings) -> None:
    """启用 Langfuse 时校验必需配置。"""

    if not settings.langfuse_tracing_enabled:
        return

    required_values = {
        "LANGFUSE_PUBLIC_KEY": settings.langfuse_public_key,
        "LANGFUSE_SECRET_KEY": settings.langfuse_secret_key,
        "LANGFUSE_BASE_URL": settings.langfuse_base_url,
    }
    missing_names = [name for name, value in required_values.items() if not value]
    if missing_names:
        raise ValueError(
            "Langfuse 已启用，但缺少配置："
            + ", ".join(missing_names)
            + "。请检查项目 .env。"
        )


def _settings_key(settings: Settings) -> tuple[str, str, str, str]:
    return (
        settings.langfuse_public_key,
        settings.langfuse_secret_key,
        settings.langfuse_base_url,
        settings.langfuse_environment,
    )


def initialize_observability(
    settings: Settings | None = None,
) -> Langfuse | None:
    """按需初始化并复用 Langfuse 客户端。"""

    global _langfuse_client, _langfuse_config_key

    settings = settings or get_settings()
    validate_langfuse_settings(settings)
    if not settings.langfuse_tracing_enabled:
        return None

    config_key = _settings_key(settings)
    if _langfuse_client is not None:
        if _langfuse_config_key != config_key:
            raise RuntimeError(
                "Langfuse 已使用另一组配置初始化，"
                "请重启服务后再切换项目或环境。"
            )
        return _langfuse_client

    _langfuse_client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
        environment=settings.langfuse_environment,
        tracing_enabled=True,
    )
    _langfuse_config_key = config_key
    return _langfuse_client


def attach_intent_tag(config: RunnableConfig, intent: str) -> None:
    """把意图补挂到已建好的跟踪配置上。

    build_run_config 在意图识别之前调用（那会儿还不知道意图），所以
    意图定下来后由这里补一个 intent:* 标签。观测页的「意图成本」卡
    靠这个标签在 Langfuse 里按意图聚合 token（见 /admin/observe）。
    """

    if not intent:
        return
    tag = f"intent:{intent}"
    tags = list(config.get("tags") or [])
    if tag not in tags:
        tags.append(tag)
    config["tags"] = tags
    metadata = config.get("metadata")
    if isinstance(metadata, dict):
        langfuse_tags = list(metadata.get("langfuse_tags") or [])
        if tag not in langfuse_tags:
            langfuse_tags.append(tag)
        metadata["langfuse_tags"] = langfuse_tags
        metadata["intent"] = intent


def build_run_config(
    *,
    thread_id: str,
    mode: str,
    settings: Settings | None = None,
) -> RunnableConfig:
    """为 LangChain/LangGraph 构造统一的跟踪配置。"""

    settings = settings or get_settings()
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "run_name": f"feishu-{mode}",
        "tags": ["feishu-paper-agent", mode],
        "metadata": {
            "langfuse_session_id": thread_id,
            "langfuse_tags": ["feishu-paper-agent", mode],
            "mode": mode,
        },
    }

    # token 计量器恒挂（不依赖 Langfuse 开关）：意图识别、改写、Agent
    # 循环、Planner 各步共用这份 config，所有 LLM 调用都会被累计。
    # 请求结束时 read_token_usage(config) 取总值，写进 /admin/logs。
    meter = TokenMeter()
    callbacks: list = [meter]
    if settings.langfuse_tracing_enabled:
        initialize_observability(settings)
        callbacks.append(CallbackHandler(public_key=settings.langfuse_public_key))
    config["callbacks"] = callbacks
    config["configurable"]["token_meter"] = meter

    return config


def flush_observability() -> None:
    """立即上报已缓冲的 trace，上报失败不影响主业务。"""

    if _langfuse_client is None:
        return

    try:
        _langfuse_client.flush()
    except Exception as exc:
        logger.warning("Langfuse trace 刷新失败：%s", exc)


def shutdown_observability() -> None:
    """关闭 Langfuse 后台上报器并释放资源。"""

    global _langfuse_client, _langfuse_config_key

    if _langfuse_client is None:
        return

    try:
        _langfuse_client.shutdown()
    except Exception as exc:
        logger.warning("Langfuse 关闭失败：%s", exc)
    finally:
        _langfuse_client = None
        _langfuse_config_key = None


def check_langfuse_connection(settings: Settings | None = None) -> bool:
    """显式检查 Langfuse 凭据，只供运维脚本调用。"""

    settings = settings or get_settings()
    if not settings.langfuse_tracing_enabled:
        raise ValueError(
            "Langfuse tracing 未启用，请在 .env 设置 "
            "LANGFUSE_TRACING_ENABLED=true。"
        )

    client = initialize_observability(settings)
    if client is None:
        return False

    authenticated = client.auth_check()
    client.flush()
    return authenticated


def get_observability_status(
    settings: Settings | None = None,
) -> dict[str, Any]:
    """返回可对外展示的安全状态，不包含凭据。"""

    settings = settings or get_settings()
    return {
        "provider": "langfuse",
        "enabled": settings.langfuse_tracing_enabled,
        "environment": settings.langfuse_environment,
    }
