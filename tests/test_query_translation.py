import pytest

from app import query_translation as qt
from app.query_translation import (
    TranslationResult,
    cjk_ratio,
    needs_translation,
    translate_to_english,
)


# ---------- 语言判断 ----------


@pytest.fixture(autouse=True)
def _clean_translation_cache():
    """每个用例前后清翻译缓存——同查询的缓存命中会让降级用例失效。"""

    qt.clear_translation_cache()
    yield
    qt.clear_translation_cache()


def test_ratio_is_one_for_pure_chinese():
    assert cjk_ratio("生计韧性") == 1.0


def test_ratio_is_zero_for_pure_english():
    assert cjk_ratio("livelihood resilience") == 0.0


def test_ratio_is_zero_for_empty_text():
    assert cjk_ratio("") == 0.0


def test_chinese_query_needs_translation():
    assert needs_translation("生计韧性怎么衡量") is True


def test_english_query_does_not_need_translation():
    assert needs_translation("How are sample counties selected?") is False


def test_mixed_query_with_few_chinese_chars_does_not_need_translation():
    """英文问题里夹个中文人名，不该触发翻译。"""

    assert needs_translation("What did the project author write about?") is False


# ---------- 翻译与降级 ----------


def test_skips_translation_for_english(monkeypatch, make_settings):
    """英文查询不该触发模型调用——翻译只对中文有意义。"""

    def fail_build_llm(_settings):
        raise AssertionError("英文查询不应调用翻译模型")

    monkeypatch.setattr(qt, "build_llm", fail_build_llm)

    result = translate_to_english(
        "How are counties selected?",
        settings=make_settings(),
    )

    assert result.translated is False
    assert result.english == "How are counties selected?"
    assert "不是中文" in result.reason


def test_translates_chinese_query(monkeypatch, make_settings):
    class FakeLlm:
        def bind(self, **kwargs):
            return self

        def invoke(self, messages, config=None):
            return type("R", (), {"content": "How is livelihood resilience measured?"})()

    monkeypatch.setattr(qt, "build_llm", lambda _settings: FakeLlm())

    result = translate_to_english("生计韧性怎么衡量", settings=make_settings())

    assert result.translated is True
    assert result.english == "How is livelihood resilience measured?"
    assert result.original == "生计韧性怎么衡量"


def test_strips_quotes_from_model_output(monkeypatch, make_settings):
    class FakeLlm:
        def bind(self, **kwargs):
            return self

        def invoke(self, messages, config=None):
            return type("R", (), {"content": '"How is it measured?"'})()

    monkeypatch.setattr(qt, "build_llm", lambda _settings: FakeLlm())

    result = translate_to_english("怎么衡量", settings=make_settings())

    assert result.english == "How is it measured?"


def test_falls_back_when_model_raises(monkeypatch, make_settings):
    class BrokenLlm:
        def bind(self, **kwargs):
            return self

        def invoke(self, messages, config=None):
            raise RuntimeError("模型接口不可用")

    monkeypatch.setattr(qt, "build_llm", lambda _settings: BrokenLlm())

    result = translate_to_english("生计韧性怎么衡量", settings=make_settings())

    assert result.translated is False
    assert result.english == "生计韧性怎么衡量"
    assert "异常" in result.reason


def test_falls_back_when_model_returns_empty(monkeypatch, make_settings):
    class EmptyLlm:
        def bind(self, **kwargs):
            return self

        def invoke(self, messages, config=None):
            return type("R", (), {"content": "   "})()

    monkeypatch.setattr(qt, "build_llm", lambda _settings: EmptyLlm())

    result = translate_to_english("生计韧性怎么衡量", settings=make_settings())

    assert result.translated is False


def test_falls_back_when_model_echoes_chinese_back(monkeypatch, make_settings):
    """模型有时把中文抄回来。那等于没翻译，BM25 照样匹配不上。"""

    class EchoLlm:
        def bind(self, **kwargs):
            return self

        def invoke(self, messages, config=None):
            return type("R", (), {"content": "生计韧性怎么衡量"})()

    monkeypatch.setattr(qt, "build_llm", lambda _settings: EchoLlm())

    result = translate_to_english("生计韧性怎么衡量", settings=make_settings())

    assert result.translated is False
    assert "仍是中文" in result.reason


def test_falls_back_when_translation_is_absurdly_long(monkeypatch, make_settings):
    """模型跑飞时会开始自问自答，输出长度会失控。"""

    class VerboseLlm:
        def bind(self, **kwargs):
            return self

        def invoke(self, messages, config=None):
            return type("R", (), {"content": "How to measure it? " * 200})()

    monkeypatch.setattr(qt, "build_llm", lambda _settings: VerboseLlm())

    result = translate_to_english("生计韧性怎么衡量", settings=make_settings())

    assert result.translated is False
    assert "长度异常" in result.reason


# ---------- 翻译缓存 ----------


class CountingLlm:
    """记录 invoke 次数的假 LLM。"""

    def __init__(self, content="How is livelihood resilience measured?"):
        self.content = content
        self.calls = 0

    def bind(self, **kwargs):
        return self

    def invoke(self, messages, config=None):
        self.calls += 1
        return type("R", (), {"content": self.content})()


def test_same_query_translated_once(monkeypatch, make_settings, _clean_translation_cache):
    llm = CountingLlm()
    monkeypatch.setattr(qt, "build_llm", lambda _settings: llm)

    first = translate_to_english("生计韧性怎么衡量", settings=make_settings())
    second = translate_to_english("生计韧性怎么衡量", settings=make_settings())

    assert llm.calls == 1, "第二次应命中缓存，不再调模型"
    assert second.english == first.english
    assert second.translated is True


def test_disabled_cache_calls_model_every_time(
    monkeypatch, make_settings, _clean_translation_cache
):
    llm = CountingLlm()
    monkeypatch.setattr(qt, "build_llm", lambda _settings: llm)

    settings = make_settings(translation_cache_enabled=False)
    translate_to_english("生计韧性怎么衡量", settings=settings)
    translate_to_english("生计韧性怎么衡量", settings=settings)

    assert llm.calls == 2


def test_failed_translation_not_cached(monkeypatch, make_settings, _clean_translation_cache):
    """降级结果（模型返回还是中文）不缓存——下次要重试，不能粘住失败。"""

    llm = CountingLlm(content="生计韧性")  # 输出仍是中文 → 降级
    monkeypatch.setattr(qt, "build_llm", lambda _settings: llm)

    first = translate_to_english("生计韧性怎么衡量", settings=make_settings())
    assert first.translated is False

    llm.content = "How is it measured?"  # 修好了
    second = translate_to_english("生计韧性怎么衡量", settings=make_settings())
    assert second.translated is True
    assert llm.calls == 2, "失败结果不进缓存，第二次真实重试"
