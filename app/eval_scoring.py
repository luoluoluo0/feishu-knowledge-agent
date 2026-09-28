import re
from dataclasses import dataclass, field
from typing import Any


# 测试集的判分逻辑。
#
# 判分只用可复现的规则，不引入第二个模型来打分。
# 原因：同一个模型既出题又判分时，它「认为答对了」的标准和它自己答题时
# 用的标准是一致的，于是真实读者看不懂的回复照样能拿高分。
# 要的是客观尺子，不是自我感觉良好。
#
# 三个层次，由硬到软：
#
#   1. 检索命中 —— 期望的块或文献在不在 top-k 里。纯集合运算，最硬。
#   2. 关键词覆盖 —— 回复里有没有这些关键事实。归一化后子串匹配，可复现。
#   3. 拒答判定 —— 该说「语料里没有」时有没有硬凑。看它有没有乱引文献。
#
# 关键词这一层是软肋：写成别的说法就算没覆盖。所以关键词由出题时一并给出，
# 且必须逐个校验「确实能在原文里找到」，把模型的自由发挥挡在外面。


# 判分前把大小写、空白、标点全部抹平。
#
# 中英文混排时标点差异很大（全角 vs 半角），不归一化会出现
# 「回复里明明写了，却被判成没写」的假阴性，而且这种假阴性
# 只在中文标点出现时发生，很难从总分上看出来。
_PUNCT_RE = re.compile(
    r"[\s　，。；：、（）【】「」『』《》〈〉“”‘’！？·—－–…"
    r",.;:()\[\]{}<>\"'`!?/\\|~@#$%^&*+=_-]+"
)


# 明确的「找不到」表述。
#
# 用模式匹配而不是固定短语表。固定的短语表永远是漏的——实测模型说过
# 「没有关于…的论文」「没有专门研究…的论文」「资料库中不存在…」，
# 这些都不在早期那张表里，结果 40 条正确拒答里有 26 条被判成失败。
#
# 判据看的是「有没有明确告诉用户找不到」，不是「有没有提到文献」。
# 正确的拒答常常会列举召回到的文献作为佐证——那是在说明「我查到了
# 这些，但都不相关」，是负责任的行为，不是硬凑。
REFUSAL_PATTERNS = (
    r"没有.{0,15}(论文|文献|资料|内容|研究|涉及|提及|收录)",
    r"未.{0,15}(找到|收录|包含|涉及|提及|查到|检索到)",
    r"不.{0,8}(存在|包含|涉及|相关)",
    r"(资料库|文献库|语料库|知识库|库里|库中).{0,15}(没有|无|不存在|未收录|未包含)",
    # 「没有/无/未」+ 短距离内的「相关」。这条是为了接住
    # 「没有发现相关文献」——上面第一条刻意不含「发现」，
    # 因为「没有发现显著影响」是研究结论，不是拒答。
    r"(没有|无|未).{0,8}相关",
    # 「不能基于资料库回答」「无法回答」这类。上面几条都要求出现
    # 「没有/未/不存在」，接不住「不能」「无法」开头的说法——
    # 实测有一条正确拒答写的是「不能基于资料库回答，命题本身也不成立」，
    # 被漏判成硬凑。
    r"(不能|无法|没法|难以).{0,8}回答",
    r"找不到|无法回答|无从回答|无法确定",
)

_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS))

# 一次判分最多认多少条命中，避免长回答靠堆砌蒙混。
MAX_KEYWORD_HITS = 99


def normalize(text: str) -> str:
    """判分前统一归一化：大小写、空白、标点全部抹平。"""

    if not text:
        return ""
    return _PUNCT_RE.sub("", str(text).lower())


def contains(haystack: str, needle: str) -> bool:
    """归一化后的子串匹配。"""

    target = normalize(needle)
    if not target:
        return False
    return target in normalize(haystack)


def keyword_coverage(answer: str, keywords: list[str]) -> tuple[int, int]:
    """回复覆盖了几个关键词。返回 (命中数, 总数)。"""

    wanted = [k for k in (keywords or []) if normalize(k)]
    if not wanted:
        return 0, 0
    hit = sum(1 for k in wanted[:MAX_KEYWORD_HITS] if contains(answer, k))
    return hit, len(wanted)


def missing_keywords(answer: str, keywords: list[str]) -> list[str]:
    """没覆盖到的关键词——报告里要列出来，才看得出错在哪。"""

    return [k for k in (keywords or []) if normalize(k) and not contains(answer, k)]


# ---------------------------------------------------------------- 检索命中


@dataclass
class RetrievalScore:
    chunk_hit: bool
    item_hit: bool
    hit_rank: int | None
    expected_items: list[str] = field(default_factory=list)
    got_items: list[str] = field(default_factory=list)


def score_retrieval(results: list[Any], case: dict) -> RetrievalScore:
    """看检索结果里有没有期望的块和文献。

    results 是带 metadata 的检索结果对象（MilvusStore 返回的那种）。
    """

    expected_chunks = set(case.get("expected_chunk_ids") or [])
    expected_items = set(case.get("expected_item_ids") or [])

    got_chunks: list[str] = []
    got_items: list[str] = []
    for item in results:
        meta = getattr(item, "metadata", None) or {}
        got_chunks.append(str(meta.get("chunk_id", "")))
        got_items.append(str(meta.get("item_id", "")))

    rank = None
    for index, chunk_id in enumerate(got_chunks):
        if chunk_id in expected_chunks:
            rank = index + 1
            break

    return RetrievalScore(
        chunk_hit=bool(expected_chunks & set(got_chunks)),
        item_hit=bool(expected_items & set(got_items)),
        hit_rank=rank,
        expected_items=sorted(expected_items),
        got_items=got_items,
    )


# ---------------------------------------------------------------- 拒答判定


def has_refusal_marker(answer: str) -> bool:
    return bool(_REFUSAL_RE.search(str(answer or "")))


_CJK_RE = re.compile(r"[一-鿿]")


def _looks_chinese(text: str) -> bool:
    text = str(text or "")
    if not text:
        return False
    return len(_CJK_RE.findall(text)) / len(text) > 0.3


def _is_language_bearing(text: str) -> bool:
    """这段文本里有没有语言成分（汉字或拉丁字母）。

    表格里的数值——3.41、(0.011)、-0.135*——既不是中文也不是英文，
    它们跨语言通用，中文回复里照样会出现。不把它们排除掉的话，
    「表 7 里 Jakarta Selatan 的实际贫困率是多少」这种题会被误判成
    跨语言题，进而走错判分分支。
    """

    return bool(re.search(r"[A-Za-z一-鿿]", str(text or "")))


def is_cross_lingual_case(case: dict) -> bool:
    """问题中文、关键事实词却是英文——典型的跨语言题。

    语料九成是英文，所以给这类题出的关键事实词必然是英文的（出题时要
    校验「能在原文里原样找到」）。但系统面向中文读者，会用中文作答。
    拿英文词去中文回复里找必然全部落空——实测 60 条里 41 条被误判成
    失败，而其中 33 条的回复其实正确引用了期望的那篇文献。
    """

    query = str(case.get("query") or "")
    keywords = [str(k) for k in (case.get("answer_keywords") or []) if str(k).strip()]
    if not keywords or not _looks_chinese(query):
        return False

    # 分母是「全部关键词」，不只看带语言的那几个——数值虽然不参与
    # 语言判断，但它确实降低了「这条题整体是跨语言的」这个判断的置信度。
    english = sum(
        1 for k in keywords if _is_language_bearing(k) and not _looks_chinese(k)
    )
    return english > len(keywords) / 2


def mentions_expected_title(
    answer: str,
    item_ids: list[str],
    title_map: dict[str, str],
    min_prefix: int = 8,
) -> str | None:
    """回复里有没有引用期望的那篇文献。命中返回 item_id。

    跨语言题改用这个判分：篇名是确定的，系统无论用中文还是英文作答，
    只要找对了文献就会带出篇名。比关键词匹配可靠得多。
    """

    normalized = normalize(answer)
    if not normalized:
        return None

    for item_id in item_ids or []:
        title = normalize(title_map.get(str(item_id), ""))
        if len(title) >= min_prefix and title[:min_prefix] in normalized:
            return str(item_id)
    return None


def cites_corpus_paper(answer: str, titles: list[str], min_prefix: int = 8) -> str | None:
    """回复里有没有引用语料中的具体文献。

    命中就返回那个标题，没命中返回 None。

    匹配用标题的前缀而不是全称：回复里通常不会把四十字的标题写全，
    但会写「《信息穷人还是信息富人》」这样的开头。前缀够长才不会
    把「信息分化」这种通用词误判成引用。
    """

    normalized = normalize(answer)
    if not normalized:
        return None

    for title in titles:
        prefix = normalize(title)
        if len(prefix) < min_prefix:
            continue
        if prefix[:min_prefix] in normalized:
            return title
    return None


# ---------------------------------------------------------------- 综合判分


@dataclass
class CaseScore:
    case_id: str
    category: str
    passed: bool
    checks: dict[str, Any] = field(default_factory=dict)

    # 没判出结果。
    #
    # 比如检索模式下拿不到最终回复，拒答题就无从判起——它判的是
    # 「有没有硬凑文献」，而那是最终回复才有的东西。
    #
    # 这种不能算通过。算进去报告会虚高：显示「40 条拒答全对」，
    # 实际一条都没判。虚高的分数比低分更坏，因为它让人放心。
    unjudged: bool = False

    def summary(self) -> str:
        state = "未判" if self.unjudged else ("过" if self.passed else "不过")
        return f"{self.case_id} [{self.category}] {state}"


def _keyword_check(answer: str, case: dict, threshold: float = 0.5) -> dict[str, Any]:
    hit, total = keyword_coverage(answer, case.get("answer_keywords") or [])
    if total == 0:
        return {"skipped": True}
    return {
        "hit": hit,
        "total": total,
        "rate": hit / total,
        "passed": hit / total >= threshold,
        "missing": missing_keywords(answer, case.get("answer_keywords") or [])[:6],
    }


def score_case(
    case: dict,
    *,
    results: list[Any] | None = None,
    answer: str | None = None,
    corpus_titles: list[str] | None = None,
    title_map: dict[str, str] | None = None,
    error: str | None = None,
) -> CaseScore:
    """给一条测试用例判分。

    results 是检索结果（检索模式），answer 是系统回复（端到端模式）。
    两者可以只给一个，也可以都给。
    """

    case_id = str(case.get("id", "?"))
    category = str(case.get("category", "?"))
    checks: dict[str, Any] = {}

    # 系统直接报错，一律算不过。边界用例也要求「不崩」。
    if error:
        checks["error"] = error
        return CaseScore(case_id, category, False, checks)

    if results is not None:
        checks["retrieval"] = score_retrieval(results, case).__dict__

    if answer is not None:
        checks["answer_len"] = len(answer)

    retrieval_passed: bool | None = None
    if results is not None and (case.get("expected_chunk_ids") or case.get("expected_item_ids")):
        score = score_retrieval(results, case)
        # 严格标准看块，宽松标准看文献。两个都记，passed 取宽松的那个，
        # 因为文献级命中已经能说明「找对方向了」，块级是排序质量问题。
        retrieval_passed = score.chunk_hit or score.item_hit
        checks["retrieval"] = score.__dict__

    if category == "refusal":
        if answer is None:
            # 没有最终回复就判不了拒答，标成未判而不是通过。
            return CaseScore(case_id, category, True, checks, unjudged=True)

        said_not_found = has_refusal_marker(answer)
        cited = cites_corpus_paper(answer, corpus_titles or [])
        checks["refusal"] = {
            "said_not_found": said_not_found,
            "cited_paper": cited,
            "passed": said_not_found,
        }
        return CaseScore(case_id, category, said_not_found, checks)

    if category == "edge":
        # 边界用例只要求不崩：有回复、没抛异常，就算过。
        passed = answer is not None and len(str(answer).strip()) > 0
        if results is not None and not (case.get("expected_chunk_ids") or case.get("expected_item_ids")):
            passed = True
        return CaseScore(case_id, category, passed, checks)

    if category == "metadata":
        if answer is None:
            return CaseScore(case_id, category, retrieval_passed is not False, checks)
        expected = case.get("expected_values") or []
        matched = [v for v in expected if contains(answer, v)]
        checks["metadata"] = {
            "expected": expected,
            "matched": matched,
            "passed": bool(expected) and len(matched) == len(expected),
        }
        return CaseScore(case_id, category, checks["metadata"]["passed"], checks)

    # 事实类、摘要类、对比类、多轮类：检索和关键词都算。
    verdicts: list[bool] = []
    if retrieval_passed is not None:
        verdicts.append(retrieval_passed)
    if answer is not None:
        if is_cross_lingual_case(case):
            # 跨语言题不能用英文关键事实词去判中文回复——那必然全部落空，
            # 所以改判「有没有引用期望的那篇文献」，篇名是确定的。
            #
            # 但篇名匹配不能**取代**关键词判据：模型有时只写编号
            # （「该文献编号 055」）而不复述篇名，也有的题压根不合适用篇名判
            # （比如问表格里某个数值）。所以两个信号取其一。
            cited = mentions_expected_title(
                answer, case.get("expected_item_ids") or [], title_map or {}
            )
            kw = _keyword_check(answer, case)
            kw_passed = bool(kw.get("passed")) if not kw.get("skipped") else False

            checks["cross_lingual"] = {
                "cited_expected_item": cited,
                "keywords": kw,
                "passed": cited is not None or kw_passed,
            }
            verdicts.append(cited is not None or kw_passed)
        else:
            kw = _keyword_check(answer, case)
            checks["keywords"] = kw
            if not kw.get("skipped"):
                verdicts.append(bool(kw["passed"]))

    passed = all(verdicts) if verdicts else False
    return CaseScore(case_id, category, passed, checks)
