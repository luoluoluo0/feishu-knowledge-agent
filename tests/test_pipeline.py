import pytest

from app import pipeline
from app.coreference import Turn
from app.intent import INTENT_WHITELIST, IntentResult
from app.pipeline import (
    PLANNER_AGENT,
    PLANNER_INTENTS,
    TOOL_AGENT,
    prepare_question,
    route_for_intent,
)
from app.coreference import RewriteResult


def make_rewrite(
    question: str = "它的方法呢？",
    rewritten: str = "第001篇文献的方法是什么？",
    item_id: str = "001",
) -> RewriteResult:
    return RewriteResult(
        original_question=question,
        rewritten_query=rewritten,
        item_id=item_id,
        rewritten=rewritten != question,
        reason="测试用。",
    )


def make_intent(question: str, intent: str = "simple_qa") -> IntentResult:
    return IntentResult(question=question, intent=intent, reason="测试用。")


def merged_payload(
    rewritten: str = "第001篇文献的方法是什么？",
    item_id: str = "001",
    intent: str = "simple_qa",
) -> str:
    """合并调用会输出的 JSON 文本。"""

    import json

    return json.dumps(
        {
            "rewritten_query": rewritten,
            "item_id": item_id,
            "intent": intent,
            "reason": "测试用。",
        },
        ensure_ascii=False,
    )


class FakeUnderstandLlm:
    """返回预设的合并 JSON，记录收到的 messages 与 config。"""

    def __init__(self, payload: str, recorded: dict):
        self.payload = payload
        self.recorded = recorded

    def invoke(self, messages, config=None, **kwargs):
        self.recorded["understand_messages"] = messages
        self.recorded["understand_config"] = config
        return type("Response", (), {"content": self.payload})()


def patch_history(monkeypatch, recorded):
    """替换 config 构造与历史加载——两条路径都要。"""

    monkeypatch.setattr(
        pipeline,
        "build_run_config",
        lambda **_: {
            "configurable": {},
            "run_name": "test",
            "metadata": {"langfuse_session_id": "s"},
        },
    )
    monkeypatch.setattr(
        pipeline, "load_history", lambda *a, **k: recorded.setdefault("history", [])
    )


def patch_understand(monkeypatch, recorded, payload: str):
    """走合并调用路径（Path B）：需要改写的场景。"""

    patch_history(monkeypatch, recorded)
    monkeypatch.setattr(
        pipeline, "build_llm", lambda _settings: FakeUnderstandLlm(payload, recorded)
    )


def patch_intent_only(monkeypatch, recorded, intent=None):
    """走单独意图调用路径（Path A）：首轮、带编号等无需改写的场景。"""

    patch_history(monkeypatch, recorded)

    def fake_intent(question, **kwargs):
        recorded["intent_question"] = question
        recorded["intent_config"] = kwargs.get("config")
        return intent or make_intent(question)

    monkeypatch.setattr(pipeline, "classify_intent", fake_intent)


def _turn_with_intent(intent: str) -> Turn:
    return Turn(
        question="对比第012篇和第045篇的研究方法",
        rewritten_query="对比第012篇和第045篇的研究方法",
        item_id="",
        answer="上一轮的对比结论……",
        intent=intent,
    )


# ---------- 路由表 ----------


def test_group_report_routes_to_planner():
    assert route_for_intent("group_report") == PLANNER_AGENT


def test_compare_papers_routes_to_planner():
    assert route_for_intent("compare_papers") == PLANNER_AGENT


@pytest.mark.parametrize(
    "intent",
    ["simple_qa", "metadata_query", "summary_paper", "summary_ppt"],
)
def test_other_intents_route_to_tool_agent(intent):
    assert route_for_intent(intent) == TOOL_AGENT


def test_unknown_intent_falls_back_to_tool_agent():
    assert route_for_intent("something_else") == TOOL_AGENT


def test_planner_intents_are_known_intents():
    """路由表里出现的意图必须在意图白名单里，否则永远不会被选中。"""

    assert PLANNER_INTENTS <= INTENT_WHITELIST


def test_summary_intents_are_known_intents():
    from app.pipeline import SUMMARY_INTENTS

    assert SUMMARY_INTENTS <= INTENT_WHITELIST


# ---------- 预处理：合并调用路径（需要改写） ----------


def test_prepare_question_returns_original_question(monkeypatch, make_settings):
    recorded = {"history": [_turn_with_intent("simple_qa")]}
    patch_understand(monkeypatch, recorded, merged_payload())

    prepared = prepare_question(
        "它的方法呢？", "t1", mode="auto", settings=make_settings()
    )

    assert prepared.original_question == "它的方法呢？"
    assert prepared.thread_id == "t1"
    assert prepared.rewrite.rewritten_query == "第001篇文献的方法是什么？"
    assert prepared.intent.intent == "simple_qa"


def test_prepare_question_classifies_the_rewritten_query(monkeypatch, make_settings):
    """意图的判定对象是改写后的查询——合并调用里由提示词内部保证。"""

    recorded = {"history": [_turn_with_intent("simple_qa")]}
    patch_understand(monkeypatch, recorded, merged_payload())

    prepared = prepare_question(
        "它的方法呢？", "t1", mode="auto", settings=make_settings()
    )

    assert prepared.intent.question == "第001篇文献的方法是什么？"


def test_prepare_question_marks_the_merged_stage(monkeypatch, make_settings):
    recorded = {"history": [_turn_with_intent("simple_qa")]}
    patch_understand(monkeypatch, recorded, merged_payload())

    prepare_question("它的方法呢？", "t1", mode="auto", settings=make_settings())

    assert recorded["understand_config"]["run_name"] == "feishu-query-understand"
    assert recorded["understand_config"]["metadata"]["stage"] == "query_understand"


def test_prepare_question_sends_history_and_question_to_the_model(
    monkeypatch, make_settings
):
    recorded = {"history": [_turn_with_intent("simple_qa")]}
    patch_understand(monkeypatch, recorded, merged_payload())

    prepare_question("它的方法呢？", "t1", mode="auto", settings=make_settings())

    human = recorded["understand_messages"][-1].content
    assert "它的方法呢？" in human
    assert "对比第012篇和第045篇的研究方法" in human


def test_prepare_question_keeps_session_metadata_in_stage_configs(
    monkeypatch, make_settings
):
    recorded = {"history": [_turn_with_intent("simple_qa")]}
    patch_understand(monkeypatch, recorded, merged_payload())

    prepare_question("它的方法呢？", "t1", mode="auto", settings=make_settings())

    assert recorded["understand_config"]["metadata"]["langfuse_session_id"] == "s"


def test_prepare_question_does_not_pollute_the_base_config(monkeypatch, make_settings):
    recorded = {"history": [_turn_with_intent("simple_qa")]}
    patch_understand(monkeypatch, recorded, merged_payload())

    prepared = prepare_question(
        "它的方法呢？", "t1", mode="auto", settings=make_settings()
    )

    # 阶段 config 是从主 config 派生的副本，不能污染主 config。
    assert "stage" not in prepared.config["metadata"]
    assert prepared.config["metadata"]["langfuse_session_id"] == "s"


# ---------- 预处理：单独意图调用路径（无需改写） ----------


def test_prepare_question_first_turn_skips_rewrite_llm_call(monkeypatch, make_settings):
    """首轮没有历史，改写本来就不花调用——只应发起意图调用。"""

    recorded = {"history": []}

    def forbidden_llm(_settings):
        raise AssertionError("首轮不应发起改写调用")

    patch_history(monkeypatch, recorded)
    monkeypatch.setattr(pipeline, "build_llm", forbidden_llm)
    monkeypatch.setattr(
        pipeline, "classify_intent", lambda q, **k: make_intent(q, "simple_qa")
    )

    prepared = prepare_question(
        "知识库里有哪些文献？", "t1", mode="auto", settings=make_settings()
    )

    assert prepared.rewrite.rewritten is False
    assert "首轮" in prepared.rewrite.reason
    assert prepared.intent.intent == "simple_qa"


def test_prepare_question_explicit_id_skips_rewrite_llm_call(
    monkeypatch, make_settings
):
    recorded = {"history": [_turn_with_intent("simple_qa")]}

    def forbidden_llm(_settings):
        raise AssertionError("带编号的问题不应发起改写调用")

    patch_history(monkeypatch, recorded)
    monkeypatch.setattr(pipeline, "build_llm", forbidden_llm)
    monkeypatch.setattr(
        pipeline, "classify_intent", lambda q, **k: make_intent(q, "summary_paper")
    )

    prepared = prepare_question(
        "第001篇用了什么数据？", "t1", mode="auto", settings=make_settings()
    )

    assert prepared.rewrite.rewritten is False
    assert prepared.intent.intent == "summary_paper"  # 意图结果原样透传


# ---------- 追问延续 ----------


def test_followup_after_task_inherits_task_intent(monkeypatch, make_settings):
    """改写过的追问 + 上一轮是任务意图 → 沿用任务意图，不降级 simple_qa。"""

    recorded = {"history": [_turn_with_intent("compare_papers")]}
    # 意图识别把「再加上013」判成 simple_qa（实测行为）
    patch_understand(
        monkeypatch,
        recorded,
        merged_payload(rewritten="对比第012篇、第045篇和第013篇的研究方法", intent="simple_qa"),
    )

    prepared = prepare_question(
        "再加上第013篇", "t1", mode="auto", settings=make_settings()
    )

    assert prepared.rewrite.rewritten is True
    assert prepared.intent.intent == "compare_papers"
    assert "沿用上一轮任务意图" in prepared.intent.reason


def test_followup_without_history_stays_simple_qa(monkeypatch, make_settings):
    recorded = {"history": []}
    patch_intent_only(monkeypatch, recorded, make_intent("再加上013", "simple_qa"))

    prepared = prepare_question(
        "再加上第013篇", "t1", mode="auto", settings=make_settings()
    )
    assert prepared.intent.intent == "simple_qa"


def test_followup_after_simple_qa_stays_simple_qa(monkeypatch, make_settings):
    """上一轮就是单发问答时没有任务可延续。"""

    recorded = {"history": [_turn_with_intent("simple_qa")]}
    patch_understand(
        monkeypatch, recorded, merged_payload(rewritten="为什么呢", intent="simple_qa")
    )

    prepared = prepare_question(
        "为什么呢", "t1", mode="auto", settings=make_settings()
    )
    assert prepared.intent.intent == "simple_qa"


def test_followup_inherits_even_if_not_rewritten(monkeypatch, make_settings):
    """「再加上013」不含代词，指代消解不会改写它——延续规则照样触发。"""

    recorded = {"history": [_turn_with_intent("compare_papers")]}
    patch_understand(
        monkeypatch,
        recorded,
        merged_payload(rewritten="再加上第013篇", intent="simple_qa"),
    )

    prepared = prepare_question(
        "再加上第013篇", "t1", mode="auto", settings=make_settings()
    )
    assert prepared.rewrite.rewritten is False
    assert prepared.intent.intent == "compare_papers"


def test_followup_without_item_id_does_not_inherit(monkeypatch, make_settings):
    """不带编号的泛泛追问（「为什么呢」）不被劫持进任务链路。

    带编号（「再加上013」）是增量的强信号；不带编号时信任意图识别。
    """

    recorded = {"history": [_turn_with_intent("compare_papers")]}
    patch_understand(
        monkeypatch, recorded, merged_payload(rewritten="为什么呢", intent="simple_qa")
    )

    prepared = prepare_question(
        "为什么呢", "t1", mode="auto", settings=make_settings()
    )
    assert prepared.intent.intent == "simple_qa"
