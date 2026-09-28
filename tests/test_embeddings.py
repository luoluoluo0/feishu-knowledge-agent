import pytest

from app.embeddings import (
    EMBED_MAX_CHARS_EN,
    EMBED_MAX_CHARS_ZH,
    RetryingEmbeddings,
    build_embeddings,
    estimate_char_limit,
)


class FakeInner:
    """模拟上游：前 fail_times 次抛 400，之后成功。"""

    def __init__(self, fail_times: int = 0):
        self.fail_times = fail_times
        self.calls = 0
        self.result = [0.1, 0.2]

    def embed_query(self, text, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(
                "Error code: 400 - {'code': 20015, "
                "'message': 'The parameter is invalid.'}"
            )
        return self.result

    def embed_documents(self, texts):
        return [self.result for _ in texts]


def test_success_needs_no_retry():
    inner = FakeInner()
    embeddings = RetryingEmbeddings(inner, attempts=3, delay=0)

    assert embeddings.embed_query("测试") == [0.1, 0.2]
    assert inner.calls == 1


def test_retries_until_success():
    inner = FakeInner(fail_times=2)
    embeddings = RetryingEmbeddings(inner, attempts=3, delay=0)

    assert embeddings.embed_query("测试") == [0.1, 0.2]
    assert inner.calls == 3


def test_raises_after_exhausting_attempts():
    inner = FakeInner(fail_times=99)
    embeddings = RetryingEmbeddings(inner, attempts=3, delay=0)

    with pytest.raises(RuntimeError, match="20015"):
        embeddings.embed_query("测试")

    assert inner.calls == 3


def test_passes_through_other_attributes():
    embeddings = RetryingEmbeddings(FakeInner(), attempts=3, delay=0)

    assert embeddings.embed_documents(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]


def test_unknown_attribute_raises_attribute_error():
    embeddings = RetryingEmbeddings(FakeInner(), attempts=3, delay=0)

    with pytest.raises(AttributeError):
        embeddings.not_a_real_attribute


def test_build_embeddings_wraps_the_client(make_settings):
    embeddings = build_embeddings(make_settings())

    assert isinstance(embeddings, RetryingEmbeddings)
    assert embeddings.model == "test-embedding"


def test_build_embeddings_requires_api_key(make_settings):
    with pytest.raises(ValueError, match="SILICON_API_KEY"):
        build_embeddings(make_settings(silicon_api_key=""))


# ---------- 长度估算 ----------


def test_pure_chinese_matches_the_chinese_endpoint():
    # 不带标点，保证每类字符占比纯粹。
    assert estimate_char_limit("这是一段纯中文的测试文本没有任何英文字符") == (
        EMBED_MAX_CHARS_ZH
    )


def test_pure_english_matches_the_english_endpoint():
    assert estimate_char_limit("ThisIsPureEnglishWithoutAnyPunctuation") == (
        EMBED_MAX_CHARS_EN
    )


def test_mixed_text_lands_between_the_two_limits():
    limit = estimate_char_limit("中文中文中文中文中文abcdefghijabcdefghij")

    assert EMBED_MAX_CHARS_ZH < limit < EMBED_MAX_CHARS_EN


def test_mixed_text_is_far_below_the_arithmetic_midpoint():
    """混排限值明显低于两端点的算术平均——这是倒数关系，不是线性关系。

    50% 中文的正确限值约 682，算术平均是 852。
    早期用线性插值算成 890，导致中英混排的块大量超限被接口拒绝
    （实测失败率接近 10%）。这条断言就是防它退化回去。
    """

    limit = estimate_char_limit("中文中文中文中文中文" + "abcdefghij")
    arithmetic_midpoint = (EMBED_MAX_CHARS_ZH + EMBED_MAX_CHARS_EN) / 2

    assert limit < arithmetic_midpoint - 100


def test_punctuation_lowers_the_limit():
    """标点会被拆成多个 token，密度越高上限越低。"""

    plain = "This is a sentence without any punctuation at all"
    punctuated = "Wagstaff, A. (2003). J. Econ., 12(3), 315-331; doi:10.1016/j."

    assert estimate_char_limit(punctuated) < estimate_char_limit(plain)


def test_chinese_limit_is_stricter_than_english():
    """中文一个字的 token 消耗约为英文一个字符的三倍，上限必须更严。"""

    assert EMBED_MAX_CHARS_ZH < EMBED_MAX_CHARS_EN


def test_empty_text_returns_a_generous_limit():
    assert estimate_char_limit("") > EMBED_MAX_CHARS_EN
