import logging
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.config import Settings, get_settings
from app.json_utils import parse_llm_json
from app.llm import build_llm


# 指代消解：把带指代或省略的追问改写成不依赖上下文也能独立理解的查询。
#
# 分工：
# - 规则只判断“不需要改写”，用来省掉模型调用。
# - 识别“需要改写”交给模型，因为中文零指代省略句（“那方法呢？”）
#   没有任何可匹配的文本特征。


# 识别问题里的文献编号，比如「第 31 篇」「031 号」。
# 旧的路由模块（已删除）里有一份同值的正则；编号写法若有扩展，
# 这里直接改这一份即可。
ITEM_ID_PATTERN = re.compile(
    r"(?:第\s*)?(\d{1,3})\s*(?:篇|号|这篇|item)",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)

# 算「任务」的意图（对比/总结/提纲）：这些任务进行中时，短编号追问
# （「加上第8篇」）是增量指令——编号只说明加什么，不说明加到哪，
# 需要模型结合历史扩写成完整任务。pipeline 的追问延续规则共用这一组。
SUMMARY_INTENTS = {
    "summary_paper",
    "summary_ppt",
    "compare_papers",
    "group_report",
}

# 改写输出长度上限：原问题长度乘以此系数再加常数。
# 超出视为模型跑飞（例如自问自答），降级回原问题。
MAX_REWRITE_RATIO = 5
MAX_REWRITE_EXTRA = 50

REWRITE_SYSTEM_PROMPT = """你是一个对话查询改写模块。
你的任务是把用户的最新问题改写成不依赖对话历史也能独立理解的完整问题。

规则：
1. 只输出 JSON，不要输出解释或其他文字。
2. 如果最新问题已经能独立理解（没有代词、没有省略），rewritten_query 原样返回该问题。
3. 如果存在指代或省略，用对话历史里的信息补全。补全时优先用具体编号表达被指代的对象，
   例如把「它」写成「第001篇文献」。
4. 只使用历史里出现过的信息，不要编造。
5. 如果历史里找不到指代对象，rewritten_query 原样返回最新问题，item_id 留空。
6. 增量/比较类指令要扩写成完整任务：如果历史里正在做某项任务
   （如对比第003篇和第022篇的研究方法），而最新问题是往里加东西
   （如「加上第8篇」），rewritten_query 要写成本来的任务加上新对象
   （如「对比第003篇、第022篇和第008篇的研究方法」），
   不能只写「加上第8篇」这种离开历史就看不懂的半句话。

JSON 格式：
{
  "rewritten_query": "改写后的完整问题",
  "item_id": "三位编号，例如 001；没有就留空字符串",
  "reason": "为什么这样改写"
}
"""


@dataclass
class Turn:
    """会话历史上的一轮，只保留改写需要的最小信息。

    intent 记录这一轮被判定的任务意图：追问延续规则要用——
    「再加上013」这类增量指令该沿用上一轮的任务型意图，
    而不是被判成 simple_qa 降级成单发问答。
    """

    question: str
    rewritten_query: str
    item_id: str
    answer: str
    intent: str = ""


@dataclass
class RewriteResult:
    """一次改写的结果。

    rewritten 恒等于 rewritten_query != original_question，
    只表示“查询文本是否变了”，不承载其他语义。
    """

    original_question: str
    rewritten_query: str
    item_id: str
    rewritten: bool
    reason: str


def normalize_item_id(value: str) -> str:
    """把 1、01、001 统一成 001。"""

    value = str(value or "").strip()
    if value.isdigit():
        return value.zfill(3)
    return value


def has_explicit_item_id(question: str) -> bool:
    """判断问题里是否已经出现明确的文献编号。"""

    return ITEM_ID_PATTERN.search(question) is not None


def needs_rewrite(question: str, history: list[Turn]) -> bool:
    """判断是否需要交给模型做改写。

    只识别“不需要改写”的情况，识别不了“需要改写”的情况，
    所以历史非空且问题里没写明编号时一律返回 True，由模型自己判断。

    例外：上一轮是任务意图（对比/总结/提纲）时，带编号的短追问
    （「加上第8篇」）不短路——编号只说明加什么，不说明加到哪，
    必须让模型结合历史扩写成完整任务（提示词里有对应的增量扩写规则）。
    """

    if not history:
        return False

    if has_explicit_item_id(question):
        if (history[-1].intent or "") in SUMMARY_INTENTS:
            return True
        return False

    return True


def _fallback(question: str, reason: str) -> RewriteResult:
    """构造一个“不改写”的结果。所有降级路径都走这里。"""

    return RewriteResult(
        original_question=question,
        rewritten_query=question,
        item_id="",
        rewritten=False,
        reason=reason,
    )


def build_history_text(history: list[Turn]) -> str:
    """把历史轮次拼成给模型看的文本。"""

    lines = []
    for index, turn in enumerate(history, start=1):
        lines.append(f"第{index}轮")
        lines.append(f"用户：{turn.question}")
        if turn.answer:
            lines.append(f"助手：{turn.answer}")
    return "\n".join(lines)


def parse_rewrite_json(text: str, question: str) -> RewriteResult:
    """解析模型输出。任何不合法的情况都回退到原问题。"""

    data = parse_llm_json(text)
    if data is None:
        return _fallback(question, "模型输出不是合法 JSON，回退到原问题。")

    if not isinstance(data, dict):
        return _fallback(question, "模型输出不是 JSON 对象，回退到原问题。")

    raw_query = data.get("rewritten_query")
    if not isinstance(raw_query, str) or not raw_query.strip():
        return _fallback(question, "模型没有给出改写结果，回退到原问题。")

    rewritten_query = raw_query.strip()
    if len(rewritten_query) > len(question) * MAX_REWRITE_RATIO + MAX_REWRITE_EXTRA:
        return _fallback(question, "模型输出长度异常，回退到原问题。")

    reason = str(data.get("reason") or "").strip()

    return RewriteResult(
        original_question=question,
        rewritten_query=rewritten_query,
        item_id=normalize_item_id(data.get("item_id", "")),
        rewritten=rewritten_query != question,
        reason=reason or "模型完成改写。",
    )


def rewrite_query(
    question: str,
    history: list[Turn],
    settings: Settings | None = None,
    config: RunnableConfig | None = None,
) -> RewriteResult:
    """把带指代的追问改写成自包含的查询。

    任何失败都不向上抛出，一律降级为“使用原问题”。
    """

    if not needs_rewrite(question, history):
        if not history:
            return _fallback(question, "历史为空，首轮无需改写。")
        return _fallback(question, "问题已含明确编号，无需改写。")

    settings = settings or get_settings()

    try:
        messages = [
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"对话历史：\n{build_history_text(history)}\n\n"
                    f"最新问题：{question}"
                )
            ),
        ]
        response = build_llm(settings).invoke(messages, config=config)
        return parse_rewrite_json(response.content, question)
    except Exception as exc:
        logger.warning("查询改写调用失败，回退到原问题：%s", exc)
        return _fallback(question, f"改写调用异常，回退到原问题：{exc}")
