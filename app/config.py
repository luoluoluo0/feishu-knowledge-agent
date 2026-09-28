import os
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


# app 目录放“运行时业务代码”。
# 这里统一读取配置，避免每个文件都自己读 .env。
PROJECT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_DIR / ".env")

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    silicon_api_key: str
    embeddings_base_url: str
    embeddings_model: str
    qdrant_url: str
    qdrant_collection: str
    milvus_uri: str
    milvus_token: str
    milvus_collection: str
    retrieval_top_k: int
    rerank_enabled: bool
    rerank_model: str
    rerank_min_score: float
    checkpoint_db_path: str
    conversation_db_path: str
    service_api_key: str
    rate_limit_per_minute: int
    context_slim_enabled: bool
    context_answer_head_chars: int
    langfuse_tracing_enabled: bool
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_base_url: str
    langfuse_environment: str
    # 单次 LLM 调用的超时秒数。不设的话 openai SDK 默认 600 秒，
    # 一次挂起的调用会占住工作线程十分钟。
    llm_timeout_seconds: int = 60
    # 命中子块后是否扩展到父块补全上下文，以及父块文本预算。
    parent_expand_enabled: bool = True
    parent_expand_max_chars: int = 1600
    # rerank 门槛按意图分档。
    #
    # 0.8 是按事实型问题校准的（语料内 0.9+，语料外最高 0.6662）。
    # 总结/对比/提纲类问题的宽泛查询 reranker 天然打低分（语料内
    # 实测 0.005~0.76），用 0.8 会系统性误拒。分档后事实类维持 0.8
    # （拒答防线不动），总结类走独立的低门槛，取值见校准数据
    # （dev-notes/ch02.md）。关掉开关即回到全局单门槛。
    rerank_threshold_by_intent: bool = True
    rerank_min_score_summary: float = 0.35
    # 进程内查询结果缓存：相同问题（归一化后）直接复用上次检索产出，
    # 省掉翻译→嵌入→检索→重排约 4 秒和全部 API 费用。TTL 防语料更新后
    # 返回旧答案；容量上限防内存膨胀。跑评测前要关掉，否则重跑会命中
    # 缓存、测不到检索层的真实变化。
    query_cache_enabled: bool = True
    query_cache_ttl_seconds: int = 3600
    query_cache_max_entries: int = 256
    # 翻译缓存：中文查询→英文 BM25 查询的确定性调用（temperature=0），
    # 同查询重翻是纯浪费（实测 566ms/次）。TTL 可比查询缓存长——
    # 翻译不依赖语料状态；降级结果不缓存，避免瞬时失败粘住。
    translation_cache_enabled: bool = True
    translation_cache_ttl_seconds: int = 86400
    translation_cache_max_entries: int = 512
    # 会话上下文窗口：历史估算 token 超过阈值时，把最旧轮次的检索原文
    # 摘要化（本轮保真），防长会话撑爆模型窗口。阈值内原样透传、前缀
    # 稳定，DeepSeek 前缀缓存照常命中——「压完定死」，与被否决的每轮
    # 精简方案（context_slim_enabled，见 app/context.py）的关键区别。
    context_window_enabled: bool = True
    context_window_max_tokens: int = 24000
    # 数据保留策略：超过保留天数的请求日志与会话轮次先归档（JSONL）再删除，
    # 不活跃线程的 LangGraph checkpoint 一并清理，防三个 SQLite 库无限增长。
    data_retention_days: int = 90
    retention_archive_dir: str = str(PROJECT_DIR / "data" / "archive")
    # 同步端点的线程池上限（anyio 默认 40）。每个 /ask 占一个线程 8~30 秒，
    # 调大只缓解排队，不改变并发保护靠限流的事实。
    app_threadpool_tokens: int = 120
    # 单次请求 token 告警阈值：超过即打 WARNING（服务端）并在前端标红。
    # 首轮全量实测均值 1.8 万、最大 6 万——5 万定在"正常组会提纲也到不了，
    # 但异常膨胀一定看得见"的位置。
    token_alert_threshold: int = 50000
    # 图表专家（自部署微调模型）：Agent 的第 11 个工具。
    # 总闸默认关闭——关闭时工具不注册、提示词不加段，主路径零变化。
    # base_url 指向本地隧道（ssh -L 18000:127.0.0.1:8000 <AutoDL>）
    # 或同机部署的 OpenAI 兼容服务。
    chart_expert_enabled: bool = False
    chart_expert_base_url: str = "http://127.0.0.1:18000/v1"
    chart_expert_model: str = "qwen2.5-7b-lora-v2"
    chart_expert_timeout: int = 60
    # 飞书共享文件夹自动同步。默认关闭，避免仅升级代码就开始访问外部资源。
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    feishu_folder_tokens: str = ""
    feishu_sync_enabled: bool = False
    feishu_event_mode: str = "websocket"
    feishu_reconcile_interval_seconds: int = 3600
    feishu_sync_db_path: str = str(PROJECT_DIR / "data" / "runtime" / "feishu_sync.db")
    feishu_allowed_types: str = "docx,pdf"
    feishu_delete_grace_days: int = 7
    feishu_worker_max_attempts: int = 5
    feishu_pdf_parser: str = "auto"
    feishu_sync_download_dir: str = str(
        PROJECT_DIR / "data" / "runtime" / "feishu_sync" / "files"
    )


def get_env(name: str, default: str = "") -> str:
    """读取环境变量，并去掉首尾空格。"""

    return os.getenv(name, default).strip()


def get_bool_env(name: str, default: bool = False) -> bool:
    """读取布尔环境变量。

    只接受常见的真假值，避免把拼写错误的配置静默当成 False。
    """

    raw_value = get_env(name, "true" if default else "false").lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} 只能是 true/false、1/0、yes/no 或 on/off，"
        f"当前值为 {raw_value!r}。"
    )


def get_service_api_key() -> str:
    current = get_env("SERVICE_API_KEY")
    legacy = get_env("FEISHU_SERVICE_API_KEY")
    if current:
        return current
    if legacy:
        logger.warning(
            "FEISHU_SERVICE_API_KEY 已弃用，请迁移到 SERVICE_API_KEY；本次仍兼容。"
        )
        return legacy
    # 安全地默认拒绝：没有显式配置时由鉴权层返回配置错误，
    # 不能让公开部署共享一个可猜到的开发密钥。
    return ""


@lru_cache
def get_settings() -> Settings:
    """读取项目配置。

    SERVICE_API_KEY 是访问 FastAPI 接口用的业务 key；旧变量
    FEISHU_SERVICE_API_KEY 仅用于兼容。未配置时受保护接口拒绝服务。
    """

    return Settings(
        # Agent 的对话模型用 DeepSeek 官方的 deepseek-flash；
        # 向量化与重排仍走 SiliconFlow（见上面的 silicon_api_key）。
        #
        # 两家分开是有意的。SiliconFlow 上也提供 DeepSeek 模型，但那边
        # 的余额是给 embedding 和 rerank 用的——混在一起的话，一边用光
        # 两边一起停。实测就踩过：SiliconFlow 余额不足时，embedding、
        # rerank、对话模型同时挂掉，整个 Agent 不可用。
        #
        # 只加载当前项目根目录的 .env，避免误读父目录里其他项目的密钥。
        deepseek_api_key=get_env("DEEPSEEK_API_KEY"),
        deepseek_base_url=get_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_model=get_env("DEEPSEEK_MODEL", "deepseek-flash"),
        silicon_api_key=get_env("SILICON_API_KEY"),
        embeddings_base_url=get_env("EMBEDDINGS_BASE_URL", "https://api.siliconflow.cn/v1"),
        embeddings_model=get_env("EMBEDDINGS_MODEL", "BAAI/bge-large-zh-v1.5"),
        qdrant_url=get_env("QDRANT_URL", "http://127.0.0.1:6333"),
        qdrant_collection=get_env("QDRANT_COLLECTION", "feishu_paper_share"),
        # Milvus 用于替换 Qdrant。本地 Docker 部署不需要 token，
        # 将来切到 Zilliz Cloud 时把 uri 和 token 填进 .env 即可。
        milvus_uri=get_env("MILVUS_URI", "http://127.0.0.1:19530"),
        milvus_token=get_env("MILVUS_TOKEN"),
        # 默认指向带 BM25 的那个集合。另一个同名集合是改造前的旧版，
        # 没有 sparse 字段，跑混合检索会逐条报错。
        milvus_collection=get_env("MILVUS_COLLECTION", "feishu_paper_child_bm25"),
        retrieval_top_k=int(get_env("RETRIEVAL_TOP_K", "5")),
        rerank_enabled=get_bool_env("RERANK_ENABLED", True),
        rerank_model=get_env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        # 重排分数的置信度门槛。
        #
        # 实测：取 0.8 时，40 条语料外的问题 100% 被拦住，155 条该命中的
        # 留住 88.4%。拒答组最高分是 0.6662，跟理论最优点 0.6761 只差
        # 0.01——那个「零误伤」很可能只是样本运气，所以往上留出余量，
        # 用 1.9pp 的命中率换 0.13 的安全边际。
        rerank_min_score=float(get_env("RERANK_MIN_SCORE", "0.8")),
        checkpoint_db_path=get_env(
            "FEISHU_CHECKPOINT_DB_PATH",
            str(PROJECT_DIR / "data" / "runtime" / "agent_checkpoints.db"),
        ),
        conversation_db_path=get_env(
            "FEISHU_CONVERSATION_DB_PATH",
            str(PROJECT_DIR / "data" / "runtime" / "conversation_history.db"),
        ),
        service_api_key=get_service_api_key(),
        rate_limit_per_minute=int(get_env("FEISHU_RATE_LIMIT_PER_MINUTE", "3")),
    # 每次调用模型之前，把历史里用过的检索原文换成一行提示。
    #
    # 默认关闭。实测开启后准确率不掉（90% vs 90%），但延迟反而从
    # 20.9 秒涨到 40.3 秒——原因见 app/context.py 的说明：每轮重拼
    # 都会改写提示前缀，把 DeepSeek 的前缀缓存整个打掉。
    # 缓存的 token 本来就又便宜又快，省它等于白省。
    context_slim_enabled=get_bool_env("CONTEXT_SLIM_ENABLED", False),
    # 历史回答保留开头多少字。太短会丢结论，太长又省不下来。
    context_answer_head_chars=int(get_env("CONTEXT_ANSWER_HEAD_CHARS", "300")),
    # 命中子块后扩展到父块补全上下文（small-to-big）。
    #
    # 子块是为了检索精度切的（段落级），但模型回答常常需要它所在的
    # 整节：子块只有一句结论，父块才有论证过程。扩展只在最终结果上
    # 做，不影响召回与重排。
    parent_expand_enabled=get_bool_env("PARENT_EXPAND_ENABLED", True),
    # 每个父块给模型看的最长字符数。父块中位约 2400 字，但存在几十万
    # 字的极端块（整本附录），不截断的话一次检索就能撑爆上下文。
    parent_expand_max_chars=int(get_env("PARENT_EXPAND_MAX_CHARS", "1600")),
        langfuse_tracing_enabled=get_bool_env(
            "LANGFUSE_TRACING_ENABLED",
            False,
        ),
        langfuse_public_key=get_env("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=get_env("LANGFUSE_SECRET_KEY"),
        langfuse_base_url=get_env(
            "LANGFUSE_BASE_URL",
            "https://cloud.langfuse.com",
        ),
        langfuse_environment=get_env(
            "LANGFUSE_TRACING_ENVIRONMENT",
            "development",
        ),
        llm_timeout_seconds=int(get_env("FEISHU_LLM_TIMEOUT_SECONDS", "60")),
        rerank_threshold_by_intent=get_bool_env("RERANK_THRESHOLD_BY_INTENT", True),
        rerank_min_score_summary=float(get_env("RERANK_MIN_SCORE_SUMMARY", "0.35")),
        query_cache_enabled=get_bool_env("QUERY_CACHE_ENABLED", True),
        query_cache_ttl_seconds=int(get_env("QUERY_CACHE_TTL_SECONDS", "3600")),
        query_cache_max_entries=int(get_env("QUERY_CACHE_MAX_ENTRIES", "256")),
        translation_cache_enabled=get_bool_env("TRANSLATION_CACHE_ENABLED", True),
        translation_cache_ttl_seconds=int(
            get_env("TRANSLATION_CACHE_TTL_SECONDS", "86400")
        ),
        translation_cache_max_entries=int(
            get_env("TRANSLATION_CACHE_MAX_ENTRIES", "512")
        ),
        context_window_enabled=get_bool_env("CONTEXT_WINDOW_ENABLED", True),
        context_window_max_tokens=int(get_env("CONTEXT_WINDOW_MAX_TOKENS", "24000")),
        data_retention_days=int(get_env("DATA_RETENTION_DAYS", "90")),
        retention_archive_dir=get_env(
            "DATA_ARCHIVE_DIR", str(PROJECT_DIR / "data" / "archive")
        ),
        app_threadpool_tokens=int(get_env("APP_THREADPOOL_TOKENS", "120")),
        token_alert_threshold=int(get_env("TOKEN_ALERT_THRESHOLD", "50000")),
        chart_expert_enabled=get_bool_env("CHART_EXPERT_ENABLED", False),
        chart_expert_base_url=get_env(
            "CHART_EXPERT_BASE_URL", "http://127.0.0.1:18000/v1"
        ),
        chart_expert_model=get_env("CHART_EXPERT_MODEL", "qwen2.5-7b-lora-v2"),
        chart_expert_timeout=int(get_env("CHART_EXPERT_TIMEOUT", "60")),
        feishu_app_id=get_env("FEISHU_APP_ID"),
        feishu_app_secret=get_env("FEISHU_APP_SECRET"),
        feishu_folder_tokens=get_env("FEISHU_FOLDER_TOKENS"),
        feishu_sync_enabled=get_bool_env("FEISHU_SYNC_ENABLED", False),
        feishu_event_mode=get_env("FEISHU_EVENT_MODE", "websocket"),
        feishu_reconcile_interval_seconds=int(
            get_env("FEISHU_RECONCILE_INTERVAL_SECONDS", "3600")
        ),
        feishu_sync_db_path=get_env(
            "FEISHU_SYNC_DB_PATH",
            str(PROJECT_DIR / "data" / "runtime" / "feishu_sync.db"),
        ),
        feishu_allowed_types=get_env("FEISHU_ALLOWED_TYPES", "docx,pdf"),
        feishu_delete_grace_days=int(get_env("FEISHU_DELETE_GRACE_DAYS", "7")),
        feishu_worker_max_attempts=int(get_env("FEISHU_WORKER_MAX_ATTEMPTS", "5")),
        feishu_pdf_parser=get_env("FEISHU_PDF_PARSER", "auto"),
        feishu_sync_download_dir=get_env(
            "FEISHU_SYNC_DOWNLOAD_DIR",
            str(PROJECT_DIR / "data" / "runtime" / "feishu_sync" / "files"),
        ),
    )
