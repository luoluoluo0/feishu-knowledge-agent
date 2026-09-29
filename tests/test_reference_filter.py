from __future__ import annotations

import dataclasses

import pytest

from app.query_cache import clear as clear_query_cache, make_key
from app.query_context import set_current_intent
from app.tools import PaperSearchTools


@pytest.fixture(autouse=True)
def _clean_state():
    """意图 ContextVar 和查询缓存都是进程级状态，用例间必须清干净。"""

    set_current_intent("")
    clear_query_cache()
    yield
    set_current_intent("")
    clear_query_cache()


class FakeRetrievalResult:
    """最小化模拟 milvus_store.RetrievalResult 的字段。"""

    def __init__(self):
        self.chunks = []
        self.top_score = 0.9
        self.confident = True
        self.reranked = True
        self.reason = ""


class FakePaperStore:
    """捕获 retrieve 收到的 include_reference，其余一概不关心。"""

    def __init__(self):
        self.captured = None

    def retrieve(self, query, top_k=None, **kwargs):
        self.captured = kwargs.get("include_reference")
        return FakeRetrievalResult()


def _tools_without_stack(query_cache_enabled: bool = False) -> PaperSearchTools:
    """绕过 __init__（真实构造要连 Milvus），只装配被测路径。"""

    from app.config import get_settings

    tools = PaperSearchTools.__new__(PaperSearchTools)
    tools.settings = dataclasses.replace(
        get_settings(), query_cache_enabled=query_cache_enabled
    )
    tools.paper_store = FakePaperStore()
    return tools


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        ("compare_papers", False),
        ("summary_paper", False),
        ("simple_qa", False),
        ("group_report", False),
        ("metadata_query", True),
        ("", True),  # 无意图（测试/脚本直调）保持旧行为
    ],
)
def test_paper_retrieve_filters_references_by_intent(intent, expected):
    tools = _tools_without_stack()
    set_current_intent(intent)

    tools._paper_retrieve("它的研究方法", 5)

    assert tools.paper_store.captured is expected


def test_cache_key_separates_reference_flag():
    """同一查询开关参考文献两种过滤结果不同，缓存键必须区分开。"""

    base = dict(query="研究方法", item_id="012", min_score=0.8, top_k=5)
    key_with = make_key(**base, include_reference=True)
    key_without = make_key(**base, include_reference=False)
    key_unset = make_key(**base)

    assert len({key_with, key_without, key_unset}) == 3
