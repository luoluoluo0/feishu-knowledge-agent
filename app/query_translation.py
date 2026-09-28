import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.config import Settings, get_settings
from app.llm import build_llm


# 把中文查询翻译成英文检索查询，用于激活英文 BM25。
#
# 为什么需要这一步：
#
# 语料九成是英文，而 BM25 是「词面匹配」——中文查询词和英文文档词
# 一个都对不上。跨语言场景下 BM25 那一路召回的全是无关结果，
# 但它照样会给这些结果排名，RRF 融合时就会把向量找到的正确结果挤下去。
# 实测跨语言精确命中率因此从 40% 掉到 26.7%。
#
# 翻译之后，英文 BM25 才真正生效。实测（30 条跨语言样本）：
#
#   中文向量                    精确命中 50.0%
#   中文向量 + 英文 BM25         精确命中 90.0%   ← 采用这个
#   中文向量 + 英文向量           精确命中 60.0%
#   中文向量 + 英文向量 + 英文 BM25 精确命中 83.3%
#
# 加英文向量反而变差，因为它和中文向量重叠度高，会稀释 BM25 这一路。


logger = logging.getLogger(__name__)

CJK_PATTERN = re.compile(r"[一-鿿]")

# 中文占比超过这个值才翻译。英文查询里偶尔夹几个中文词不需要翻译。
CJK_RATIO_THRESHOLD = 0.15

# 翻译结果过长视为模型跑飞（正常翻译不会比原文长这么多）。
MAX_EXPANSION_RATIO = 4.0
MAX_EXPANSION_EXTRA = 80

TRANSLATION_PROMPT = """把下面的中文问题翻译成一句英文检索查询。

要求：
1. 只输出这句英文，不要解释、不要加引号。
2. 用学术论文里常见的表达，专业术语要用领域内的标准译法。
   例如「生计韧性」译成 livelihood resilience，不要逐字直译。
3. 保持一句话，不要拆成多个问题。
"""


@dataclass
class TranslationResult:
    """一次翻译的结果。

    translated 表示是否真的产出了不同的英文查询。
    translated=False 时 english 等于原文，调用方应当放弃 BM25 那一路。
    """

    original: str
    english: str
    translated: bool
    reason: str


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(CJK_PATTERN.findall(text)) / len(text)


def needs_translation(query: str) -> bool:
    """判断查询是否需要翻译。

    只有中文为主的查询需要。英文查询原样用即可，
    中英混排里少量中文（比如英文问题里夹个中文人名）也不用翻。
    """

    return cjk_ratio(query) >= CJK_RATIO_THRESHOLD


def _fallback(query: str, reason: str) -> TranslationResult:
    return TranslationResult(
        original=query,
        english=query,
        translated=False,
        reason=reason,
    )


# ----------------------------------------------------------------------
# 翻译缓存。翻译是 temperature=0 的确定性调用，同一查询的结果永远一样；
# 每次检索都重翻一遍是纯浪费（实测单次约 566ms）。进程内 TTL + LRU，
# 与 query_cache 同款设计。只缓存真正翻译成功的请求——降级（异常/
# 长度异常/仍是中文）可能是瞬时的，粘住会让 BM25 那一路一直残废。
# ----------------------------------------------------------------------

_translation_lock = threading.Lock()
_translation_cache: "OrderedDict[str, tuple[float, TranslationResult]]" = OrderedDict()


def _cache_get(key: str, settings: Settings) -> TranslationResult | None:
    with _translation_lock:
        item = _translation_cache.get(key)
        if item is None:
            return None
        stored_at, result = item
        if time.monotonic() - stored_at > settings.translation_cache_ttl_seconds:
            _translation_cache.pop(key, None)
            return None
        _translation_cache.move_to_end(key)
        return result


def _cache_put(key: str, result: TranslationResult, settings: Settings) -> None:
    with _translation_lock:
        _translation_cache[key] = (time.monotonic(), result)
        while len(_translation_cache) > settings.translation_cache_max_entries:
            _translation_cache.popitem(last=False)


def clear_translation_cache() -> None:
    """清空翻译缓存。测试隔离用。"""

    with _translation_lock:
        _translation_cache.clear()


def translate_to_english(
    query: str,
    settings: Settings | None = None,
    config: RunnableConfig | None = None,
) -> TranslationResult:
    """把中文查询翻译成英文。

    成功的翻译结果进缓存（translation_cache_* 配置，TTL+LRU）：
    翻译是确定性调用，同查询重翻是纯浪费——一次检索 566ms，
    多工具调用的请求会重复付好几次。

    任何失败都降级为「没翻译」，由调用方决定怎么退。
    """

    if not needs_translation(query):
        return _fallback(query, "查询不是中文，无需翻译。")

    settings = settings or get_settings()
    cache_key = "t\x1f" + " ".join(query.split())

    if settings.translation_cache_enabled:
        cached = _cache_get(cache_key, settings)
        if cached is not None:
            return cached

    try:
        messages = [
            SystemMessage(content=TRANSLATION_PROMPT),
            HumanMessage(content=query),
        ]
        # 翻译要的是确定性，同一个查询每次应得到同样的结果。
        llm = build_llm(settings).bind(temperature=0)
        response = llm.invoke(messages, config=config)
        english = str(response.content).strip().strip('"').strip("'")
    except Exception as exc:
        logger.warning("查询翻译失败，BM25 那一路将跳过：%s", exc)
        return _fallback(query, f"翻译调用异常：{exc}")

    if not english:
        return _fallback(query, "模型没有给出翻译结果。")

    limit = len(query) * MAX_EXPANSION_RATIO + MAX_EXPANSION_EXTRA
    if len(english) > limit:
        return _fallback(query, "翻译结果长度异常，疑似模型跑飞。")

    # 模型有时会把中文抄回来。那等于没翻译，BM25 照样匹配不上。
    if cjk_ratio(english) >= CJK_RATIO_THRESHOLD:
        return _fallback(query, "翻译结果仍是中文，按未翻译处理。")

    result = TranslationResult(
        original=query,
        english=english,
        translated=english != query,
        reason="已翻译为英文检索查询。",
    )
    if settings.translation_cache_enabled:
        _cache_put(cache_key, result, settings)
    return result
