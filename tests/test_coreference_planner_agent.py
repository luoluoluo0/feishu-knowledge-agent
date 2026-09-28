from types import SimpleNamespace

import pytest

from app import planner_agent as planner_agent_module
from app.coreference import RewriteResult
from app.intent import IntentResult
from app.pipeline import PreparedQuestion
from app.planner import PlanResult


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
    monkeypatch.setattr(
        planner_agent_module, "prepare_question", lambda *a, **k: prepared
    )
    monkeypatch.setattr(
        planner_agent_module,
        "append_turn",
        lambda thread_id, turn, **k: recorded.update({"turn": turn}),
    )


def build_agent(monkeypatch, prepared, recorded):
    patch_pipeline(monkeypatch, prepared, recorded)

    class FakePlanner:
        def plan(self, question, config=None, history=None):
            recorded["plan_question"] = question
            recorded["plan_config"] = config
            recorded["plan_history"] = history
            return PlanResult(task_type="simple_qa", steps=[], reason="test")

    class FakeLlm:
        def invoke(self, prompt, config=None):
            recorded["final_prompt"] = prompt
            recorded["llm_config"] = config
            return SimpleNamespace(content="这是回答。")

    agent = planner_agent_module.PlannerAgent.__new__(planner_agent_module.PlannerAgent)
    agent.settings = SimpleNamespace()
    agent.planner = FakePlanner()
    agent.llm = FakeLlm()
    agent.tools = SimpleNamespace()
    return agent


def test_planner_uses_rewritten_query_for_planning(monkeypatch):
    recorded = {}
    agent = build_agent(
        monkeypatch,
        make_prepared(
            "它的研究方法是什么？", "第001篇文献的研究方法是什么？", "001"
        ),
        recorded,
    )

    agent.answer("它的研究方法是什么？", thread_id="t1")

    assert recorded["plan_question"] == "第001篇文献的研究方法是什么？"


def test_planner_uses_rewritten_query_for_final_answer(monkeypatch):
    recorded = {}
    agent = build_agent(
        monkeypatch,
        make_prepared(
            "它的研究方法是什么？", "第001篇文献的研究方法是什么？", "001"
        ),
        recorded,
    )

    agent.answer("它的研究方法是什么？", thread_id="t1")

    assert "第001篇文献的研究方法是什么？" in recorded["final_prompt"]


def test_planner_reuses_the_prepared_config_for_both_model_calls(monkeypatch):
    recorded = {}
    prepared = make_prepared(
        "它的研究方法是什么？", "第001篇文献的研究方法是什么？", "001"
    )
    agent = build_agent(monkeypatch, prepared, recorded)

    agent.answer("它的研究方法是什么？", thread_id="t1")

    assert recorded["plan_config"] is prepared.config
    assert recorded["llm_config"] is prepared.config


def test_planner_stores_original_question_in_history(monkeypatch):
    recorded = {}
    agent = build_agent(
        monkeypatch,
        make_prepared(
            "它的研究方法是什么？", "第001篇文献的研究方法是什么？", "001"
        ),
        recorded,
    )

    agent.answer("它的研究方法是什么？", thread_id="t1")

    assert recorded["turn"].question == "它的研究方法是什么？"
    assert recorded["turn"].rewritten_query == "第001篇文献的研究方法是什么？"
    assert recorded["turn"].answer == "这是回答。"


def test_planner_does_not_reprepare_when_given_prepared(monkeypatch):
    recorded = {}
    monkeypatch.setattr(
        planner_agent_module,
        "prepare_question",
        lambda *a, **k: pytest.fail("已经传了 prepared，不应再次预处理"),
    )
    monkeypatch.setattr(planner_agent_module, "append_turn", lambda *a, **k: None)

    agent = planner_agent_module.PlannerAgent.__new__(planner_agent_module.PlannerAgent)
    agent.settings = SimpleNamespace()
    agent.planner = SimpleNamespace(
        plan=lambda question, config=None, history=None: PlanResult(
            task_type="simple_qa", steps=[], reason="test"
        )
    )
    agent.llm = SimpleNamespace(
        invoke=lambda prompt, config=None: SimpleNamespace(content="这是回答。")
    )
    agent.tools = SimpleNamespace()

    agent.answer(
        "它的研究方法是什么？",
        thread_id="t1",
        prepared=make_prepared(
            "它的研究方法是什么？", "第001篇文献的研究方法是什么？", "001"
        ),
    )
