import types

import pytest

from app import query_cache
from app.milvus_store import RetrievedChunk


@pytest.fixture(autouse=True)
def _clean_cache():
    query_cache.clear()
    yield
    query_cache.clear()


def _patch_settings(monkeypatch, make_settings, **overrides):
    """让 query_cache 内部读到测试配置（它通过模块级 get_settings 引用）。"""

    monkeypatch.setattr(
        "app.query_cache.get_settings", lambda: make_settings(**overrides)
    )


class TestMakeKey:
    def test_whitespace_normalized(self):
        assert query_cache.make_key("第  057篇  用了什么方法") == query_cache.make_key(
            "第 057篇 用了什么方法"
        )

    def test_different_query_different_key(self):
        assert query_cache.make_key("问题A") != query_cache.make_key("问题B")

    def test_item_id_in_key(self):
        assert query_cache.make_key("同一问题", item_id="057") != query_cache.make_key(
            "同一问题", item_id="031"
        )

    def test_min_score_in_key(self):
        assert query_cache.make_key("同一问题", min_score=0.8) != query_cache.make_key(
            "同一问题", min_score=0.35
        )

    def test_corpus_revision_in_key(self):
        assert query_cache.make_key(
            "同一问题", corpus_revision=1
        ) != query_cache.make_key("同一问题", corpus_revision=2)


class TestGetPut:
    def test_roundtrip(self):
        key = query_cache.make_key("问题")
        query_cache.put(key, ("结果",))
        assert query_cache.get(key) == ("结果",)

    def test_miss_returns_none(self):
        assert query_cache.get(query_cache.make_key("没存过")) is None

    def test_expiry(self, monkeypatch, make_settings):
        _patch_settings(monkeypatch, make_settings, query_cache_ttl_seconds=1)

        key = query_cache.make_key("会过期的问题")
        query_cache.put(key, "旧结果")

        import time as time_module

        real = time_module.monotonic()
        # 时间快进 2 秒（> TTL 1 秒），条目应被视为过期。
        monkeypatch.setattr(
            "app.query_cache.time",
            types.SimpleNamespace(monotonic=lambda: real + 2),
        )
        assert query_cache.get(key) is None


class TestEviction:
    def test_lru_eviction(self, monkeypatch, make_settings):
        _patch_settings(monkeypatch, make_settings, query_cache_max_entries=2)
        keys = [query_cache.make_key(f"问题{i}") for i in range(3)]
        for i, key in enumerate(keys):
            query_cache.put(key, f"结果{i}")

        # 容量 2：最早写入的问题0 被淘汰。
        assert query_cache.get(keys[0]) is None
        assert query_cache.get(keys[1]) == "结果1"
        assert query_cache.get(keys[2]) == "结果2"

    def test_access_refreshes_lru_order(self, monkeypatch, make_settings):
        _patch_settings(monkeypatch, make_settings, query_cache_max_entries=2)
        k0, k1, k2 = (query_cache.make_key(f"问题{i}") for i in range(3))
        query_cache.put(k0, "结果0")
        query_cache.put(k1, "结果1")
        query_cache.get(k0)  # 访问 k0，让它变成「最近使用」
        query_cache.put(k2, "结果2")

        # 现在最久未用的是 k1，被淘汰的应该是它。
        assert query_cache.get(k1) is None
        assert query_cache.get(k0) == "结果0"
        assert query_cache.get(k2) == "结果2"


class TestToolsIntegration:
    """_paper_retrieve 挂接缓存：同查询只检索一次，键含 item_id。"""

    def _make_tools(self, make_settings, **overrides):
        from app.tools import PaperSearchTools

        calls = {"n": 0}

        class FakeStore:
            def retrieve(self, query, top_k=None, **kwargs):
                calls["n"] += 1
                from app.milvus_store import RetrievalResult

                chunk = RetrievedChunk(score=0.9, text="片段", metadata={"title": "t"})
                return RetrievalResult(
                    chunks=[chunk], top_score=0.9, confident=True, reranked=True, reason="ok"
                )

        tools = PaperSearchTools.__new__(PaperSearchTools)
        tools.settings = make_settings
        tools.paper_store = FakeStore()
        return tools, calls

    def test_same_query_retrieves_once(self, make_settings, monkeypatch):
        _patch_settings(monkeypatch, make_settings)
        tools, calls = self._make_tools(make_settings())

        first = tools._paper_retrieve("第057篇用了什么方法", 5)
        second = tools._paper_retrieve("第057篇用了什么方法", 5)

        assert calls["n"] == 1
        assert first == second

    def test_different_item_id_retrieves_separately(self, make_settings, monkeypatch):
        _patch_settings(monkeypatch, make_settings)
        tools, calls = self._make_tools(make_settings())

        tools._paper_retrieve("研究方法是什么", 5, item_id="057")
        tools._paper_retrieve("研究方法是什么", 5, item_id="031")

        assert calls["n"] == 2

    def test_disabled_flag_bypasses_cache(self, make_settings, monkeypatch):
        settings = make_settings(query_cache_enabled=False)
        _patch_settings(monkeypatch, settings)
        tools, calls = self._make_tools(settings)

        tools._paper_retrieve("同一个问题", 5)
        tools._paper_retrieve("同一个问题", 5)

        assert calls["n"] == 2
