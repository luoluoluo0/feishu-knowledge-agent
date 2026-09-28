import pytest

from app.config import Settings


def build_settings(**overrides) -> Settings:
    """构造测试用 Settings，默认不指向任何真实外部资源。"""

    values = {
        "deepseek_api_key": "test-deepseek-key",
        "deepseek_base_url": "https://example.test",
        "deepseek_model": "test-model",
        "silicon_api_key": "test-silicon-key",
        "embeddings_base_url": "https://example.test/v1",
        "embeddings_model": "test-embedding",
        "qdrant_url": "http://127.0.0.1:6333",
        "qdrant_collection": "test-collection",
        "milvus_uri": "http://127.0.0.1:19530",
        "milvus_token": "",
        "milvus_collection": "test-milvus-collection",
        "retrieval_top_k": 5,
        "rerank_enabled": True,
        "rerank_model": "test-rerank-model",
        "rerank_min_score": 0.8,
        # 跟生产默认值保持一致（开启）。分档相关测试自己 override。
        "rerank_threshold_by_intent": True,
        "rerank_min_score_summary": 0.35,
        # 落在 data/runtime/（已被 gitignore）：相对路径会污染仓库根目录，
        # 每次跑测试都在 cwd 留一个 test-conversation.db。
        "checkpoint_db_path": "data/runtime/test-checkpoints.db",
        "conversation_db_path": "data/runtime/test-conversation.db",
        "service_api_key": "test-service-key",
        "rate_limit_per_minute": 3,
        # 跟生产默认值保持一致。缓存/窗口/精简相关测试自己 override。
        "context_slim_enabled": False,
        "context_answer_head_chars": 300,
        "query_cache_enabled": True,
        "query_cache_ttl_seconds": 3600,
        "query_cache_max_entries": 256,
        "context_window_enabled": True,
        "context_window_max_tokens": 24000,
        # 父块扩展默认开启（与生产一致）。扩展相关测试自己 override。
        "parent_expand_enabled": True,
        "parent_expand_max_chars": 1600,
        "langfuse_tracing_enabled": False,
        "langfuse_public_key": "",
        "langfuse_secret_key": "",
        "langfuse_base_url": "https://cloud.langfuse.com",
        "langfuse_environment": "test",
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def make_settings():
    return build_settings
