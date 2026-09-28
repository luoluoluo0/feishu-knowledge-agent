import pytest

from app.query_context import (
    SUMMARY_INTENTS,
    get_current_intent,
    min_score_for_intent,
    set_current_intent,
)


@pytest.fixture(autouse=True)
def _clean_intent():
    """每个用例前后清掉 ContextVar，避免用例间串扰。"""

    set_current_intent("")
    yield
    set_current_intent("")


class TestContextVar:
    def test_default_is_empty(self):
        assert get_current_intent() == ""

    def test_set_then_get(self):
        set_current_intent("summary_paper")
        assert get_current_intent() == "summary_paper"

    def test_none_intent_normalized_to_empty(self):
        set_current_intent(None)
        assert get_current_intent() == ""

    def test_overwrite(self):
        set_current_intent("simple_qa")
        set_current_intent("group_report")
        assert get_current_intent() == "group_report"


class TestMinScoreForIntent:
    def test_fact_intents_use_default_threshold(self, make_settings):
        settings = make_settings()
        for intent in ("simple_qa", "metadata_query", ""):
            assert min_score_for_intent(intent, settings) == settings.rerank_min_score

    def test_summary_intents_use_summary_threshold(self, make_settings):
        settings = make_settings()
        for intent in SUMMARY_INTENTS:
            assert min_score_for_intent(intent, settings) == (
                settings.rerank_min_score_summary
            )

    def test_empty_intent_falls_back_to_default(self, make_settings):
        settings = make_settings()
        assert min_score_for_intent("", settings) == settings.rerank_min_score

    def test_feature_flag_disabled_unifies_thresholds(self, make_settings):
        settings = make_settings(rerank_threshold_by_intent=False)
        for intent in SUMMARY_INTENTS:
            assert min_score_for_intent(intent, settings) == settings.rerank_min_score


class FakePaperStore:
    """捕获 retrieve 调用参数的假 MilvusStore。"""

    def __init__(self):
        self.kwargs = None

    def retrieve(self, query, top_k=None, **kwargs):
        from app.milvus_store import RetrievalResult, RetrievedChunk

        self.kwargs = {"top_k": top_k, **kwargs}
        chunk = RetrievedChunk(score=0.5, text="内容片段", metadata={"title": "t"})
        return RetrievalResult(
            chunks=[chunk], top_score=0.5, confident=True, reranked=True, reason="测试"
        )


def _make_tools(settings):
    """绕过 __init__（真实 Milvus/BM25 初始化），只装配置和假 store。"""

    from app.tools import PaperSearchTools

    tools = PaperSearchTools.__new__(PaperSearchTools)
    tools.settings = settings
    tools.paper_store = FakePaperStore()
    return tools


class TestPaperRetrieveAppliesIntentThreshold:
    def teardown_method(self):
        set_current_intent("")

    def test_summary_intent_passes_summary_threshold(self, make_settings):
        set_current_intent("compare_papers")
        tools = _make_tools(make_settings())
        tools._paper_retrieve("对比两篇论文", 5)
        assert tools.paper_store.kwargs["min_score"] == 0.35

    def test_fact_intent_passes_default_threshold(self, make_settings):
        set_current_intent("metadata_query")
        tools = _make_tools(make_settings())
        tools._paper_retrieve("第031篇谁整理的", 5)
        assert tools.paper_store.kwargs["min_score"] == 0.8

    def test_no_intent_passes_default_threshold(self, make_settings):
        tools = _make_tools(make_settings())
        tools._paper_retrieve("普通问题", 5)
        assert tools.paper_store.kwargs["min_score"] == 0.8

    def test_flag_off_passes_default_threshold_even_for_summary(self, make_settings):
        set_current_intent("group_report")
        tools = _make_tools(make_settings(rerank_threshold_by_intent=False))
        tools._paper_retrieve("组会提纲", 5)
        assert tools.paper_store.kwargs["min_score"] == 0.8
