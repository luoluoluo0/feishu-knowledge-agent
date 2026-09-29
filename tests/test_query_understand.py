from app import pipeline
from app.coreference import Turn
from app.pipeline import parse_understand_json, understand_question


def test_parse_full_payload_produces_both_results():
    rewrite, intent = parse_understand_json(
        '{"rewritten_query": "第001篇文献的方法是什么？", "item_id": "1", '
        '"intent": "summary_paper", "reason": "补全指代"}',
        "它的方法呢？",
    )

    assert rewrite.rewritten_query == "第001篇文献的方法是什么？"
    assert rewrite.rewritten is True
    assert rewrite.item_id == "001"  # 1 → 001 归一化
    assert intent.intent == "summary_paper"
    assert intent.question == "第001篇文献的方法是什么？"  # 意图基于改写结果


def test_parse_invalid_json_falls_back_on_both_fields():
    rewrite, intent = parse_understand_json("这不是JSON", "它的方法呢？")

    assert rewrite.rewritten_query == "它的方法呢？"
    assert rewrite.rewritten is False
    assert intent.intent == "simple_qa"
    assert intent.question == "它的方法呢？"


def test_parse_missing_rewrite_keeps_intent():
    """改写缺失只拖垮改写侧，意图照常生效。"""

    rewrite, intent = parse_understand_json(
        '{"intent": "metadata_query"}', "有哪些文献缺PDF？"
    )

    assert rewrite.rewritten_query == "有哪些文献缺PDF？"
    assert intent.intent == "metadata_query"


def test_parse_invalid_intent_keeps_rewrite():
    """意图不在白名单只拖垮意图侧，改写照常生效。"""

    rewrite, intent = parse_understand_json(
        '{"rewritten_query": "第001篇的方法？", "intent": "hallucinated"}',
        "它的方法呢？",
    )

    assert rewrite.rewritten_query == "第001篇的方法？"
    assert intent.intent == "simple_qa"


def test_parse_oversized_rewrite_falls_back():
    """改写长度超出上限（5倍原长+50）视为异常，回退原问题。"""

    long_rewrite = "长" * 200
    rewrite, _ = parse_understand_json(
        f'{{"rewritten_query": "{long_rewrite}", "intent": "simple_qa"}}',
        "短问题",
    )

    assert rewrite.rewritten_query == "短问题"
    assert rewrite.rewritten is False
    assert "长度异常" in rewrite.reason


def test_understand_question_path_b_sends_history_and_question(monkeypatch):
    captured = {}

    class FakeLlm:
        def invoke(self, messages, config=None, **kwargs):
            captured["messages"] = messages
            return type("Response", (), {"content": '{"rewritten_query": "Q", "intent": "simple_qa"}'})()

    monkeypatch.setattr(pipeline, "build_llm", lambda _s: FakeLlm())
    history = [
        Turn(question="对比第012篇和第045篇", rewritten_query="对比第012篇和第045篇", item_id="", answer="结论")
    ]

    rewrite, intent = understand_question("它的数据呢？", history)

    human = captured["messages"][-1].content
    assert "它的数据呢？" in human
    assert "对比第012篇和第045篇" in human
    assert rewrite.rewritten_query == "Q"
    assert intent.intent == "simple_qa"


def test_understand_question_exception_degrades_both_fields(monkeypatch):
    class BrokenLlm:
        def invoke(self, messages, config=None, **kwargs):
            raise RuntimeError("网络超时")

    monkeypatch.setattr(pipeline, "build_llm", lambda _s: BrokenLlm())
    history = [Turn(question="上一轮", rewritten_query="上一轮", item_id="", answer="")]

    rewrite, intent = understand_question("它的方法呢？", history)

    assert rewrite.rewritten_query == "它的方法呢？"
    assert intent.intent == "simple_qa"
    assert "异常" in rewrite.reason


def test_understand_question_stage_config_is_derived_not_shared(monkeypatch):
    """合并调用的阶段 config 是副本，不能把 stage 写回主 config。"""

    captured = {}

    class FakeLlm:
        def invoke(self, messages, config=None, **kwargs):
            captured["config"] = config
            return type("Response", (), {"content": '{"rewritten_query": "Q", "intent": "simple_qa"}'})()

    monkeypatch.setattr(pipeline, "build_llm", lambda _s: FakeLlm())
    base_config = {"metadata": {"langfuse_session_id": "s"}}

    understand_question(
        "它的方法呢？",
        [Turn(question="上轮", rewritten_query="上轮", item_id="", answer="")],
        config=base_config,
    )

    assert captured["config"]["metadata"]["stage"] == "query_understand"
    assert "stage" not in base_config["metadata"]
