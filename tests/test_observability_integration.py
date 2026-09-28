from types import SimpleNamespace

from app import agent as agent_module
from app import pipeline as pipeline_module
from app import planner_agent as planner_agent_module
from app.planner import PlanResult


def patch_preprocessing(monkeypatch, expected_config):
    """让真实的 prepare_question 跑起来，但用固定的 config 和假的模型调用。

    只替换 config 构造与两个预处理阶段，不替换 prepare_question 本身，
    这样 prepared.config 到实际模型调用的传递链路仍被真实覆盖。
    """

    monkeypatch.setattr(
        pipeline_module, "build_run_config", lambda **_: expected_config
    )
    monkeypatch.setattr(pipeline_module, "load_history", lambda *a, **k: [])
    monkeypatch.setattr(
        pipeline_module,
        "rewrite_query",
        lambda question, history, **k: SimpleNamespace(
            original_question=question,
            rewritten_query=question,
            item_id="",
            rewritten=False,
            reason="测试跳过。",
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "classify_intent",
        lambda question, **k: SimpleNamespace(
            question=question, intent="simple_qa", reason="测试跳过。"
        ),
    )


def test_tool_agent_passes_observability_config(monkeypatch):
    expected_config = {
        "configurable": {"thread_id": "thread-tool"},
        "callbacks": ["handler"],
    }
    patch_preprocessing(monkeypatch, expected_config)
    monkeypatch.setattr(agent_module, "append_turn", lambda *a, **k: None)

    class FakeAgent:
        def invoke(self, inputs, config):
            assert inputs["messages"][0].content == "测试问题"
            assert config is expected_config
            return {"messages": []}

    agent_module.run_agent(
        FakeAgent(),
        question="测试问题",
        thread_id="thread-tool",
    )


def test_planner_agent_reuses_config_for_both_model_calls(monkeypatch, make_settings):
    expected_config = {
        "configurable": {"thread_id": "thread-planner"},
        "callbacks": ["handler"],
    }
    patch_preprocessing(monkeypatch, expected_config)
    monkeypatch.setattr(planner_agent_module, "append_turn", lambda *a, **k: None)

    class FakePlanner:
        def __init__(self):
            self.config = None

        def plan(self, question, config=None, history=None):
            self.config = config
            return PlanResult(task_type="simple_qa", steps=[], reason="test")

    class FakeLlm:
        def __init__(self):
            self.config = None

        def invoke(self, prompt, config=None):
            self.config = config
            return SimpleNamespace(content="测试回答")

    planner_agent = planner_agent_module.PlannerAgent.__new__(
        planner_agent_module.PlannerAgent
    )
    planner_agent.settings = make_settings()
    planner_agent.planner = FakePlanner()
    planner_agent.llm = FakeLlm()
    planner_agent.tools = SimpleNamespace()

    result = planner_agent.answer(
        "测试问题",
        thread_id="thread-planner",
    )

    assert result["answer"] == "测试回答"
    assert planner_agent.planner.config is expected_config
    assert planner_agent.llm.config is expected_config
