import pytest

from app.eval_scoring import (
    CaseScore,
    cites_corpus_paper,
    contains,
    has_refusal_marker,
    keyword_coverage,
    missing_keywords,
    normalize,
    score_case,
    score_retrieval,
)


class FakeHit:
    """冒充 MilvusStore 返回的检索结果。"""

    def __init__(self, chunk_id: str, item_id: str):
        self.metadata = {"chunk_id": chunk_id, "item_id": item_id}
        self.text = ""
        self.score = 0.0


# ---------- 归一化 ----------


def test_normalize_folds_case_and_space():
    assert normalize("Livelihood  Resilience") == "livelihoodresilience"


def test_normalize_folds_fullwidth_punctuation():
    """中文标点和英文标点要归一到同一个结果，否则判分有假阴性。"""

    assert normalize("生计韧性，怎么衡量？") == normalize("生计韧性, 怎么衡量?")


def test_normalize_handles_none_and_empty():
    assert normalize("") == ""
    assert normalize(None) == ""


# ---------- 关键词匹配 ----------


def test_contains_matches_across_punctuation():
    assert contains("生计韧性（Livelihood Resilience）怎么衡量", "livelihood resilience")


def test_contains_rejects_empty_needle():
    """空关键词不该匹配一切——那会让覆盖率虚高。"""

    assert contains("随便什么内容", "") is False


def test_keyword_coverage_counts_hits():
    answer = "样本来自六个县，采用双重差分方法。"
    hit, total = keyword_coverage(answer, ["六个县", "双重差分", "面板数据"])
    assert (hit, total) == (2, 3)


def test_keyword_coverage_ignores_empty_keywords():
    assert keyword_coverage("随便", []) == (0, 0)
    assert keyword_coverage("随便", ["", "  "]) == (0, 0)


def test_missing_keywords_lists_only_absent_ones():
    answer = "样本来自六个县。"
    assert missing_keywords(answer, ["六个县", "双重差分"]) == ["双重差分"]


# ---------- 检索判分 ----------


def test_score_retrieval_finds_exact_chunk():
    results = [FakeHit("001_c0003", "001"), FakeHit("017_c0001", "017")]
    score = score_retrieval(results, {"expected_chunk_ids": ["017_c0001"], "expected_item_ids": ["017"]})
    assert score.chunk_hit is True
    assert score.item_hit is True
    assert score.hit_rank == 2


def test_score_retrieval_separates_chunk_and_item_hits():
    """找对了文献但没找对块——两个指标必须分开，否则看不出排序质量问题。"""

    results = [FakeHit("017_c0009", "017")]
    score = score_retrieval(results, {"expected_chunk_ids": ["017_c0001"], "expected_item_ids": ["017"]})
    assert score.chunk_hit is False
    assert score.item_hit is True


def test_score_retrieval_handles_empty_results():
    score = score_retrieval([], {"expected_chunk_ids": ["x"], "expected_item_ids": ["y"]})
    assert score.chunk_hit is False
    assert score.item_hit is False
    assert score.hit_rank is None


# ---------- 拒答判定 ----------


@pytest.mark.parametrize(
    "text",
    [
        "语料库中没有找到相关文献。",
        # 下面这几条都是实跑里真实出现过的说法。早期用固定短语表时
        # 一条都认不出来，40 条正确拒答被判成 26 条失败。
        "这个文献库里没有关于区块链共识机制的论文。",
        "没有专门研究推荐系统冷启动的论文。",
        "资料库中没有关于该主题的任何文献。",
        "文献库中不存在相关研究。",
        "未查到这个方向的研究。",
        "没有发现相关文献。",
        "无法回答这个问题。",
    ],
)
def test_refusal_marker_covers_real_phrasings(text):
    assert has_refusal_marker(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "这篇文献采用了双重差分方法。",
        # 「没有发现」后面跟的是研究结论而不是文献，不该当成拒答。
        "研究发现数字经济发展对收入不平等没有显著影响。",
        "研究没有发现显著的性别差异。",
        "样本量为 2018 户，覆盖六个县。",
    ],
)
def test_refusal_marker_absent_in_normal_answer(text):
    assert has_refusal_marker(text) is False


def test_cites_corpus_paper_matches_title_prefix():
    """回复里通常写不全四十字的标题，只写开头。"""

    titles = ["信息穷人还是信息富人：可行能力视角下农村居民信息分化及政府支持的效应研究"]
    cited = cites_corpus_paper("参考了《信息穷人还是信息富人》这篇文献。", titles)
    assert cited == titles[0]


def test_cites_corpus_paper_ignores_short_titles():
    """标题太短会误判——「信息分化」这种词到处都是，不算引用。"""

    assert cites_corpus_paper("本文讨论信息分化问题。", ["信息分化"]) is None


def test_cites_corpus_paper_returns_none_for_empty_answer():
    assert cites_corpus_paper("", ["信息穷人还是信息富人：可行能力视角"]) is None


# ---------- 综合判分 ----------


def _case(**kwargs) -> dict:
    base = {
        "id": "eval-0001",
        "category": "en_fact",
        "query": "How were counties selected?",
        "expected_chunk_ids": ["017_c0001"],
        "expected_item_ids": ["017"],
        "answer_keywords": ["six counties", "random sampling"],
    }
    base.update(kwargs)
    return base


def test_error_fails_any_case():
    case = _case(category="edge")
    score = score_case(case, answer="whatever", error="ConnectionError: 挂了")
    assert score.passed is False
    assert "ConnectionError" in score.checks["error"]


def test_fact_case_passes_when_retrieval_and_keywords_both_hit():
    score = score_case(
        _case(),
        results=[FakeHit("017_c0001", "017")],
        answer="The study selected six counties using random sampling.",
    )
    assert score.passed is True
    assert score.checks["keywords"]["hit"] == 2


def test_fact_case_fails_when_keywords_missing():
    """检索对了但答案没答到点上——关键词这一层就是用来抓这个的。"""

    score = score_case(
        _case(),
        results=[FakeHit("017_c0001", "017")],
        answer="The paper discusses several things in general.",
    )
    assert score.passed is False
    assert "six counties" in score.checks["keywords"]["missing"]


def test_refusal_case_passes_when_it_says_not_found():
    case = _case(category="refusal", answer_keywords=[], expected_chunk_ids=[], expected_item_ids=[])
    score = score_case(
        case,
        answer="语料库中没有找到关于区块链共识机制的文献。",
        corpus_titles=["信息穷人还是信息富人：可行能力视角下农村居民"],
    )
    assert score.passed is True


def test_refusal_case_passes_even_when_it_lists_related_papers():
    """说了「找不到」之后列举召回到的文献作佐证，是负责任的行为，不是硬凑。

    判据曾经是「只要引了语料里的文献就算失败」，结果把这种正确拒答
    大批判成了幻觉。真正该看的是结论，不是有没有提到文献。
    """

    case = _case(category="refusal", answer_keywords=[], expected_chunk_ids=[], expected_item_ids=[])
    score = score_case(
        case,
        answer="未找到直接相关的文献。召回到了《信息穷人还是信息富人》等几篇，但都不相关。",
        corpus_titles=["信息穷人还是信息富人：可行能力视角下农村居民"],
    )
    assert score.passed is True
    assert score.checks["refusal"]["cited_paper"] is not None


def test_refusal_case_fails_when_it_answers_without_saying_not_found():
    """没告诉用户「找不到」，直接拿不相关的文献作答——这才是硬凑。"""

    case = _case(category="refusal", answer_keywords=[], expected_chunk_ids=[], expected_item_ids=[])
    score = score_case(
        case,
        answer="根据《信息穷人还是信息富人》一文，区块链共识机制的关键在于……",
        corpus_titles=["信息穷人还是信息富人：可行能力视角下农村居民"],
    )
    assert score.passed is False
    assert score.checks["refusal"]["said_not_found"] is False


def test_edge_case_passes_when_system_survives():
    case = _case(category="edge", answer_keywords=[], expected_chunk_ids=[], expected_item_ids=[])
    assert score_case(case, answer="没太理解你的问题。").passed is True


def test_edge_case_fails_on_empty_answer():
    case = _case(category="edge", answer_keywords=[], expected_chunk_ids=[], expected_item_ids=[])
    assert score_case(case, answer="   ").passed is False


def test_metadata_case_requires_all_values():
    case = _case(
        category="metadata",
        expected_values=["安科强"],
        answer_keywords=[],
        expected_chunk_ids=[],
        expected_item_ids=[],
    )
    assert score_case(case, answer="这篇是安科强读的。").passed is True
    assert score_case(case, answer="这篇是某人读的。").passed is False


def test_score_tolerates_case_without_expected_ids():
    """没有期望块的事实题不该直接判失败——只判关键词。"""

    case = _case(expected_chunk_ids=[], expected_item_ids=[])
    score = score_case(case, answer="six counties were selected by random sampling")
    assert score.passed is True


def test_summary_is_readable():
    score = CaseScore("eval-0007", "refusal", False)
    assert "eval-0007" in score.summary()
    assert "不过" in score.summary()
