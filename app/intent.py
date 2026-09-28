import logging
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from typing import TYPE_CHECKING

from app.config import Settings, get_settings

if TYPE_CHECKING:  # 仅类型注解用，运行时不导入（避免与 pipeline 循环）
    from app.conversation_store import Turn
from app.json_utils import parse_llm_json
from app.llm import build_llm


# 意图识别：判断用户的这次提问属于哪一类任务。
#
# 与指代消解的分工：
# - 指代消解先把追问改写成自包含查询（app/coreference.py）。
# - 本模块再对改写后的查询做分类；同时接收会话历史——追问是不是
#   上一轮任务的增量（「再加上013」），只有看着历史才分得清。
#
# 本模块只产出意图，不改变 Agent 的检索或回答行为。


logger = logging.getLogger(__name__)

INTENT_WHITELIST = {
    "simple_qa",
    "metadata_query",
    "summary_paper",
    "summary_ppt",
    "group_report",
    "compare_papers",
}

DEFAULT_INTENT = "simple_qa"

INTENT_SYSTEM_PROMPT = """你是一个飞书知识库 Agent 的意图识别模块。
你的任务是判断用户这次提问属于哪一类任务。

可选意图，只能选一个：
1. simple_qa：通用知识库问答。包括制度、流程、产品、项目、业务事实，以及不需要固定结构的普通文档问题。
2. metadata_query：元数据或统计查询。问文献编号、整理者、DOI、附件、PDF/PPT 缺失情况、文献清单、数量统计。
3. summary_paper：总结某篇论文。要论文的研究问题、方法、数据、结论、贡献。
4. summary_ppt：总结某篇 PPT 或汇报内容。
5. group_report：生成组会汇报提纲。需要按汇报顺序组织。
6. compare_papers：对比多篇文献。要相同点、不同点、适用场景。

判断规则：
- 只输出 JSON，不要输出解释或其他文字。
- 只选一个意图。
- 无法判断时返回 simple_qa。
- 出现「有哪些」「哪些文献」「缺 PDF」「缺 PPT」「谁整理的」「统计」这类问法，优先考虑 metadata_query。
- 对比两位整理者的阅读清单（「某某和某某看的论文有什么不同」「谁的阅读方向偏什么」）是清单/元数据问题，选 metadata_query——它要的是完整文献列表，不是论文正文。
- 只有对比具体论文的研究内容（方法、数据、结论的异同）才选 compare_papers。
- 出现「对比」「比较」「区别」「异同」且落在具体论文的内容上，考虑 compare_papers。
- 出现「提纲」「组会」「汇报」这类问法，考虑 group_report。
- 追问延续：如果提供了对话历史，且当前问题是上一轮任务的增量指令
  （例如上一轮在对比 012 和 045，现在说「再加上013」「把 013 也比进去」
  「那 PPT 呢」），应输出上一轮的任务类型，而不是 simple_qa 或 metadata_query。

JSON 格式：
{
  "intent": "simple_qa | metadata_query | summary_paper | summary_ppt | group_report | compare_papers",
  "reason": "为什么判断成这一类"
}
"""


@dataclass
class IntentResult:
    """一次意图识别的结果。

    question 是本次分类实际依据的文本。调用方传入的是指代消解之后的
    查询，不一定是用户原话。
    """

    question: str
    intent: str
    reason: str


def _fallback(question: str, reason: str) -> IntentResult:
    """构造一个默认意图的结果。所有降级路径都走这里。"""

    return IntentResult(
        question=question,
        intent=DEFAULT_INTENT,
        reason=reason,
    )


def parse_intent_json(text: str, question: str) -> IntentResult:
    """解析模型输出。任何不合法的情况都回退到默认意图。"""

    data = parse_llm_json(text)
    if data is None:
        return _fallback(question, "模型输出不是合法 JSON，回退到 simple_qa。")

    if not isinstance(data, dict):
        return _fallback(question, "模型输出不是 JSON 对象，回退到 simple_qa。")

    raw_intent = data.get("intent")
    if not isinstance(raw_intent, str) or not raw_intent.strip():
        return _fallback(question, "模型没有给出意图，回退到 simple_qa。")

    intent = raw_intent.strip()
    if intent not in INTENT_WHITELIST:
        return _fallback(
            question,
            f"模型给出的意图 {intent!r} 不在允许列表内，回退到 simple_qa。",
        )

    reason = str(data.get("reason") or "").strip()

    return IntentResult(
        question=question,
        intent=intent,
        reason=reason or "模型完成意图识别。",
    )


def classify_intent(
    question: str,
    settings: Settings | None = None,
    config: RunnableConfig | None = None,
    history: list["Turn"] | None = None,
) -> IntentResult:
    """判断一次提问属于哪类任务。

    传进来的 question 应当是指代消解之后的查询。history 是会话历史
    （不含本轮）——追问是不是上一轮任务的增量，只有看着历史才分得清
    （「再加上013」孤零零看会被判成 simple_qa 或 metadata_query）。

    任何失败都不向上抛出，一律降级为 simple_qa。
    """

    settings = settings or get_settings()

    context_text = question
    history_lines = [
        f"- 用户：{(turn.question or '').strip()}"
        for turn in (history or [])
        if (turn.question or "").strip()
    ]
    if history_lines:
        context_text = (
            "对话历史（最近的在后，供判断当前问题是否为上一轮任务的延续）：\n"
            + "\n".join(history_lines)
            + f"\n\n当前用户问题：{question}"
        )

    try:
        messages = [
            SystemMessage(content=INTENT_SYSTEM_PROMPT),
            HumanMessage(content=f"用户问题：{context_text}"),
        ]
        response = build_llm(settings).invoke(messages, config=config)
        return parse_intent_json(response.content, question)
    except Exception as exc:
        logger.warning("意图识别调用失败，回退到 simple_qa：%s", exc)
        return _fallback(question, f"意图识别调用异常，回退到 simple_qa：{exc}")
