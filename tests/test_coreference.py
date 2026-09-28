import json
from types import SimpleNamespace

from app import coreference
from app.coreference import (
    Turn,
    has_explicit_item_id,
    needs_rewrite,
    normalize_item_id,
)


def make_turn(question: str = "第一篇讲了什么？") -> Turn:
    return Turn(
        question=question,
        rewritten_query=question,
        item_id="001",
        answer="第001篇是《……》。",
    )


# ---------- 规则判断 ----------


def test_normalize_item_id_pads_to_three_digits():
    assert normalize_item_id("1") == "001"
    assert normalize_item_id("01") == "001"
    assert normalize_item_id("001") == "001"


def test_normalize_item_id_passes_through_non_numeric():
    assert normalize_item_id("") == ""
    assert normalize_item_id("abc") == "abc"


def test_has_explicit_item_id_matches_common_phrasings():
    assert has_explicit_item_id("第001篇讲了什么")
    assert has_explicit_item_id("001篇的方法")
    assert has_explicit_item_id("第 1 篇的结论")
    assert has_explicit_item_id("1号文献的作者")


def test_has_explicit_item_id_rejects_pronoun_only_question():
    assert not has_explicit_item_id("它的研究方法是什么？")
    assert not has_explicit_item_id("那方法呢？")


def test_needs_rewrite_is_false_for_first_turn():
    assert needs_rewrite("第一篇讲了什么？", []) is False


def test_needs_rewrite_is_false_when_question_already_has_item_id():
    history = [make_turn()]
    assert needs_rewrite("第002篇用了什么数据？", history) is False


def test_needs_rewrite_is_true_for_pronoun_followup():
    history = [make_turn()]
    assert needs_rewrite("它的研究方法是什么？", history) is True


def test_needs_rewrite_is_true_for_elliptical_followup():
    history = [make_turn()]
    assert needs_rewrite("那结论呢？", history) is True


# ---------- 历史文本拼装 ----------


def test_build_history_text_renders_turns():
    history = [
        Turn(
            question="第一篇讲了什么？",
            rewritten_query="",
            item_id="001",
            answer="第001篇是《……》。",
        ),
        Turn(question="它的方法呢？", rewritten_query="", item_id="001", answer=""),
    ]

    text = coreference.build_history_text(history)

    assert "第1轮" in text
    assert "用户：第一篇讲了什么？" in text
    assert "助手：第001篇是《……》。" in text
    assert "第2轮" in text
    assert "用户：它的方法呢？" in text


# ---------- JSON 解析与校验 ----------


def test_parse_rewrite_json_reads_valid_payload():
    payload = json.dumps(
        {
            "rewritten_query": "第001篇文献的研究方法是什么？",
            "item_id": "1",
            "reason": "把「它」还原成第001篇。",
        }
    )

    result = coreference.parse_rewrite_json(payload, "它的研究方法是什么？")

    assert result.rewritten_query == "第001篇文献的研究方法是什么？"
    assert result.item_id == "001"
    assert result.rewritten is True
    assert result.original_question == "它的研究方法是什么？"


def test_parse_rewrite_json_marks_unchanged_as_not_rewritten():
    payload = json.dumps(
        {
            "rewritten_query": "它的研究方法是什么？",
            "item_id": "",
            "reason": "已经能独立理解。",
        }
    )

    result = coreference.parse_rewrite_json(payload, "它的研究方法是什么？")

    assert result.rewritten is False
    assert result.rewritten_query == "它的研究方法是什么？"


def test_parse_rewrite_json_falls_back_on_invalid_json():
    result = coreference.parse_rewrite_json("这不是 JSON", "它的研究方法是什么？")

    assert result.rewritten is False
    assert result.rewritten_query == "它的研究方法是什么？"
    assert "JSON" in result.reason


def test_parse_rewrite_json_falls_back_on_empty_rewrite():
    payload = json.dumps({"rewritten_query": "   ", "item_id": "001", "reason": ""})

    result = coreference.parse_rewrite_json(payload, "它的研究方法是什么？")

    assert result.rewritten is False
    assert result.rewritten_query == "它的研究方法是什么？"


def test_parse_rewrite_json_falls_back_on_overlong_rewrite():
    question = "它的研究方法是什么？"
    payload = json.dumps(
        {
            "rewritten_query": "第001篇" + "很长的补充说明" * 200,
            "item_id": "001",
            "reason": "",
        }
    )

    result = coreference.parse_rewrite_json(payload, question)

    assert result.rewritten is False
    assert result.rewritten_query == question


def test_parse_rewrite_json_falls_back_when_rewritten_query_is_not_text():
    payload = json.dumps({"rewritten_query": ["a", "b"], "item_id": "", "reason": ""})

    result = coreference.parse_rewrite_json(payload, "它的研究方法是什么？")

    assert result.rewritten is False


# ---------- 改写入口与降级 ----------


def test_rewrite_query_skips_model_when_history_is_empty(monkeypatch, make_settings):
    def fail_build_llm(_settings):
        raise AssertionError("历史为空时不应调用模型")

    monkeypatch.setattr(coreference, "build_llm", fail_build_llm)

    result = coreference.rewrite_query("第一篇讲了什么？", [], settings=make_settings())

    assert result.rewritten is False
    assert result.rewritten_query == "第一篇讲了什么？"
    assert "首轮" in result.reason


def test_rewrite_query_skips_model_when_question_has_item_id(
    monkeypatch, make_settings
):
    def fail_build_llm(_settings):
        raise AssertionError("问题已含编号时不应调用模型")

    monkeypatch.setattr(coreference, "build_llm", fail_build_llm)
    history = [
        Turn(question="第一篇讲了什么？", rewritten_query="", item_id="001", answer="")
    ]

    result = coreference.rewrite_query(
        "第002篇用了什么数据？", history, settings=make_settings()
    )

    assert result.rewritten is False
    assert "编号" in result.reason


def test_rewrite_query_calls_model_and_returns_rewrite(monkeypatch, make_settings):
    payload = json.dumps(
        {
            "rewritten_query": "第001篇文献的研究方法是什么？",
            "item_id": "001",
            "reason": "补全指代。",
        }
    )

    class FakeLlm:
        def invoke(self, messages, config=None):
            return SimpleNamespace(content=payload)

    monkeypatch.setattr(coreference, "build_llm", lambda _settings: FakeLlm())
    history = [
        Turn(
            question="第一篇讲了什么？",
            rewritten_query="",
            item_id="001",
            answer="第001篇是《……》。",
        )
    ]

    result = coreference.rewrite_query(
        "它的研究方法是什么？", history, settings=make_settings()
    )

    assert result.rewritten is True
    assert result.rewritten_query == "第001篇文献的研究方法是什么？"
    assert result.item_id == "001"


def test_rewrite_query_falls_back_when_model_raises(monkeypatch, make_settings):
    class BrokenLlm:
        def invoke(self, messages, config=None):
            raise RuntimeError("模型接口不可用")

    monkeypatch.setattr(coreference, "build_llm", lambda _settings: BrokenLlm())
    history = [
        Turn(question="第一篇讲了什么？", rewritten_query="", item_id="001", answer="")
    ]

    result = coreference.rewrite_query(
        "它的研究方法是什么？", history, settings=make_settings()
    )

    assert result.rewritten is False
    assert result.rewritten_query == "它的研究方法是什么？"
