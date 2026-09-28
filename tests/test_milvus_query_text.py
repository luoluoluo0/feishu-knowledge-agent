from app.embeddings import estimate_char_limit
from app.milvus_store import to_query_text


# 查询侧的截断。
#
# 这组用例是评测集跑出来的：edge 类里有两条超长输入（1600 字和 6600 字），
# 每次都以 embedding 400 告终。文档侧一直有收缩重试保护，查询侧却是裸调用。
# embedding 模型有 512 token 硬上限，超了就是每次都拒，重试三次也一样。


def test_short_query_passes_through():
    assert to_query_text("生计韧性怎么衡量") == "生计韧性怎么衡量"


def test_folds_whitespace():
    assert to_query_text("  生计\n韧性  ") == "生计 韧性"


def test_empty_query_becomes_a_space():
    """空串会被 embedding 接口拒掉，给个空格占位。"""

    assert to_query_text("") == " "
    assert to_query_text("   ") == " "
    assert to_query_text(None) == " "


def test_long_chinese_query_is_truncated():
    query = "生计韧性" * 400
    result = to_query_text(query)

    assert len(result) < len(query)
    assert len(result) <= estimate_char_limit(query)


def test_long_english_query_is_truncated():
    query = "livelihood resilience " * 300
    result = to_query_text(query)

    assert len(result) < len(query)
    assert len(result) <= estimate_char_limit(query)


def test_medium_query_is_not_touched():
    """不该误伤正常长度的查询——截断只在超限时发生。"""

    query = "生计韧性" * 80
    if len(query) <= estimate_char_limit(query):
        assert to_query_text(query) == query


def test_mixed_language_query_gets_its_own_limit():
    """中英混排的字符上限跟纯文本不同，必须按构成算而不是拍一个数。"""

    query = "生计韧性 livelihood resilience " * 100
    result = to_query_text(query)

    assert len(result) <= estimate_char_limit(query)
