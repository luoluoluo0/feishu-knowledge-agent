from types import SimpleNamespace

import pytest

from app import agent as agent_module
from app.coreference import RewriteResult
from app.intent import IntentResult
from app.pipeline import PreparedQuestion


def make_prepared(
    question: str,
    rewritten_query: str = "",
    item_id: str = "",
    intent: str = "simple_qa",
) -> PreparedQuestion:
    rewritten = rewritten_query or question
    return PreparedQuestion(
        thread_id="t1",
        original_question=question,
        rewrite=RewriteResult(
            original_question=question,
            rewritten_query=rewritten,
            item_id=item_id,
            rewritten=bool(rewritten_query),
            reason="测试用。",
        ),
        intent=IntentResult(question=rewritten, intent=intent, reason="测试用。"),
        config={"configurable": {}, "run_name": "test", "metadata": {}},
        settings=SimpleNamespace(),
    )


def patch_pipeline(monkeypatch, prepared, recorded):
    """只替换预处理与历史写入，保留 run_agent 的编排逻辑。"""

    monkeypatch.setattr(agent_module, "prepare_question", lambda *a, **k: prepared)
    monkeypatch.setattr(
        agent_module,
        "append_turn",
        lambda thread_id, turn, **k: recorded.update(
            {"thread_id": thread_id, "turn": turn}
        ),
    )


class RecordingAgent:
    def __init__(self, recorded):
        self.recorded = recorded

    def invoke(self, inputs, config):
        self.recorded["sent"] = inputs["messages"][0].content
        self.recorded["config"] = config
        return {"messages": [SimpleNamespace(content="这是回答。")]}


def test_run_agent_sends_rewritten_query_to_agent(monkeypatch):
    recorded = {}
    prepared = make_prepared(
        "它的研究方法是什么？", "第001篇文献的研究方法是什么？", "001"
    )
    patch_pipeline(monkeypatch, prepared, recorded)

    result = agent_module.run_agent(
        RecordingAgent(recorded), question="它的研究方法是什么？", thread_id="t1"
    )

    assert recorded["sent"] == "第001篇文献的研究方法是什么？"
    assert result["rewrite"].item_id == "001"
    assert result["intent"].intent == "simple_qa"


def test_run_agent_sends_original_question_when_rewrite_skipped(monkeypatch):
    recorded = {}
    patch_pipeline(monkeypatch, make_prepared("它的研究方法是什么？"), recorded)

    agent_module.run_agent(
        RecordingAgent(recorded), question="它的研究方法是什么？", thread_id="t1"
    )

    assert recorded["sent"] == "它的研究方法是什么？"


def test_run_agent_stores_original_question_in_history(monkeypatch):
    recorded = {}
    prepared = make_prepared(
        "它的研究方法是什么？", "第001篇文献的研究方法是什么？", "001"
    )
    patch_pipeline(monkeypatch, prepared, recorded)

    agent_module.run_agent(
        RecordingAgent(recorded), question="它的研究方法是什么？", thread_id="t1"
    )

    assert recorded["turn"].question == "它的研究方法是什么？"
    assert recorded["turn"].rewritten_query == "第001篇文献的研究方法是什么？"
    assert recorded["turn"].answer == "这是回答。"
    assert recorded["thread_id"] == "t1"


def test_run_agent_uses_the_prepared_config(monkeypatch):
    recorded = {}
    prepared = make_prepared("第一篇讲了什么")
    patch_pipeline(monkeypatch, prepared, recorded)

    agent_module.run_agent(
        RecordingAgent(recorded), question="第一篇讲了什么", thread_id="t1"
    )

    assert recorded["config"] is prepared.config


def test_run_agent_does_not_reprepare_when_given_prepared(monkeypatch):
    recorded = {}
    monkeypatch.setattr(
        agent_module,
        "prepare_question",
        lambda *a, **k: pytest.fail("已经传了 prepared，不应再次预处理"),
    )
    monkeypatch.setattr(agent_module, "append_turn", lambda *a, **k: None)

    agent_module.run_agent(
        RecordingAgent(recorded),
        question="它的研究方法是什么？",
        thread_id="t1",
        prepared=make_prepared(
            "它的研究方法是什么？", "第001篇文献的研究方法是什么？", "001"
        ),
    )

    assert recorded["sent"] == "第001篇文献的研究方法是什么？"
