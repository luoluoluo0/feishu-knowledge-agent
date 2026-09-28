from dataclasses import replace

import app.resources as resources
from app.agent import build_langchain_tools
from app.config import get_settings


EXPECTED_TOOLS = {
    "search_all",
    "list_literature_by_reader",
    "semantic_search_literature",
    "list_literature_by_status",
    "get_literature_by_item_id",
    "list_missing_files",
    "hybrid_search_literature_card",
    "search_paper",
    "hybrid_search_ppt",
    "search_by_item",
    "chart_expert_answer",
}


class FakeMilvusClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakePaperStore:
    def __init__(self):
        self.client = FakeMilvusClient()


class FakeStack:
    """替代 PaperSearchTools 的轻量假栈，记录构造次数。"""

    construction_count = 0

    def __init__(self):
        FakeStack.construction_count += 1
        self.paper_store = FakePaperStore()


def _install_fake_stack(monkeypatch) -> None:
    FakeStack.construction_count = 0
    monkeypatch.setattr(resources, "PaperSearchTools", FakeStack)


def _real_stack_available() -> bool:
    """真栈构建要 15 秒且依赖 Milvus，单测里绝不能意外触发。"""

    return resources.get_retrieval_stack.cache_info().currsize == 0


def setup_function(_):
    resources.get_retrieval_stack.cache_clear()


def teardown_function(_):
    resources.get_retrieval_stack.cache_clear()


def test_singleton_returns_same_instance(monkeypatch):
    _install_fake_stack(monkeypatch)
    first = resources.get_retrieval_stack()
    second = resources.get_retrieval_stack()
    assert first is second
    assert FakeStack.construction_count == 1


def test_close_closes_milvus_client_and_clears_cache(monkeypatch):
    _install_fake_stack(monkeypatch)
    stack = resources.get_retrieval_stack()

    resources.close_retrieval_stack()

    assert stack.paper_store.client.closed is True
    assert resources.get_retrieval_stack.cache_info().currsize == 0


def test_close_is_idempotent(monkeypatch):
    _install_fake_stack(monkeypatch)

    resources.close_retrieval_stack()
    resources.close_retrieval_stack()

    assert FakeStack.construction_count == 0  # 从未构建过也能安全调用


def test_close_then_get_rebuilds(monkeypatch):
    _install_fake_stack(monkeypatch)
    resources.get_retrieval_stack()  # 第 1 次构造
    resources.close_retrieval_stack()

    resources.get_retrieval_stack()  # 关闭后重建

    assert FakeStack.construction_count == 2  # 初始 1 次 + 重建 1 次


def test_build_langchain_tools_accepts_injection():
    class StubTools:
        def __getattr__(self, name):
            # 所有工具方法都返回固定文本即可，包装层只关心可调用。
            return lambda *args, **kwargs: "stub"

    tools = build_langchain_tools(
        settings=replace(get_settings(), chart_expert_enabled=True),
        paper_tools=StubTools(),
    )

    assert len(tools) == 11
    assert {tool.name for tool in tools} == EXPECTED_TOOLS


def test_unit_tests_never_warm_real_stack():
    """防呆：单例缓存为空，说明以上测试没把真栈（15s/466MB）建出来。"""

    assert _real_stack_available()
