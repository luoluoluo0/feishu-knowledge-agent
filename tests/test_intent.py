import json
from types import SimpleNamespace

from app import intent as intent_module
from app.intent import (
    DEFAULT_INTENT,
    INTENT_SYSTEM_PROMPT,
    INTENT_WHITELIST,
    classify_intent,
    parse_intent_json,
)


# ---------- 提示词与白名单一致性 ----------


def test_whitelist_has_expected_intents():
    assert INTENT_WHITELIST == {
        "simple_qa",
        "metadata_query",
        "summary_paper",
        "summary_ppt",
        "group_report",
        "compare_papers",
    }


def test_prompt_lists_every_whitelisted_intent():
    """提示词里必须提到每一个允许的意图，避免改了一处忘了另一处。"""

    for name in INTENT_WHITELIST:
        assert name in INTENT_SYSTEM_PROMPT


def test_default_intent_is_whitelisted():
    assert DEFAULT_INTENT in INTENT_WHITELIST


# ---------- JSON 解析与校验 ----------


def test_parse_intent_json_reads_valid_payload():
    payload = json.dumps(
        {"intent": "metadata_query", "reason": "问的是资料缺失情况。"}
    )

    result = parse_intent_json(payload, "哪些文献缺 PDF？")

    assert result.intent == "metadata_query"
    assert result.reason == "问的是资料缺失情况。"
    assert result.question == "哪些文献缺 PDF？"


def test_parse_intent_json_trims_whitespace():
    payload = json.dumps({"intent": "  compare_papers  ", "reason": ""})

    result = parse_intent_json(payload, "对比这两篇")

    assert result.intent == "compare_papers"


def test_parse_intent_json_falls_back_on_invalid_json():
    result = parse_intent_json("这不是 JSON", "对比这两篇")

    assert result.intent == DEFAULT_INTENT
    assert "JSON" in result.reason


def test_parse_intent_json_falls_back_on_non_dict():
    result = parse_intent_json(json.dumps(["simple_qa"]), "对比这两篇")

    assert result.intent == DEFAULT_INTENT


def test_parse_intent_json_falls_back_on_missing_intent():
    result = parse_intent_json(json.dumps({"reason": "不知道"}), "对比这两篇")

    assert result.intent == DEFAULT_INTENT


def test_parse_intent_json_falls_back_on_non_string_intent():
    result = parse_intent_json(json.dumps({"intent": ["simple_qa"]}), "对比这两篇")

    assert result.intent == DEFAULT_INTENT


def test_parse_intent_json_rejects_value_outside_whitelist():
    payload = json.dumps({"intent": "总结", "reason": "模型自作主张用了中文。"})

    result = parse_intent_json(payload, "总结一下这篇")

    assert result.intent == DEFAULT_INTENT
    assert "总结" in result.reason


# ---------- 入口与降级 ----------


def test_classify_intent_returns_model_result(monkeypatch, make_settings):
    payload = json.dumps({"intent": "group_report", "reason": "要汇报提纲。"})

    class FakeLlm:
        def invoke(self, messages, config=None):
            return SimpleNamespace(content=payload)

    monkeypatch.setattr(intent_module, "build_llm", lambda _settings: FakeLlm())

    result = classify_intent("给我一份组会汇报提纲", settings=make_settings())

    assert result.intent == "group_report"


def test_classify_intent_falls_back_when_model_raises(monkeypatch, make_settings):
    class BrokenLlm:
        def invoke(self, messages, config=None):
            raise RuntimeError("模型接口不可用")

    monkeypatch.setattr(intent_module, "build_llm", lambda _settings: BrokenLlm())

    result = classify_intent("给我一份组会汇报提纲", settings=make_settings())

    assert result.intent == DEFAULT_INTENT
    assert "异常" in result.reason
