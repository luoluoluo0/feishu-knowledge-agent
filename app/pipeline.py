import logging
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.config import Settings, get_settings
from app.citations import reset_citation_registry
from app.conversation_store import load_history
from app.coreference import (
    ITEM_ID_PATTERN,
    MAX_REWRITE_EXTRA,
    MAX_REWRITE_RATIO,
    SUMMARY_INTENTS,
    RewriteResult,
    build_history_text,
    needs_rewrite,
    normalize_item_id,
)
from app.intent import (
    DEFAULT_INTENT,
    INTENT_WHITELIST,
    IntentResult,
    classify_intent,
)
from app.json_utils import parse_llm_json
from app.llm import build_llm
from app.observability import attach_intent_tag, build_run_config
from app.query_context import set_current_intent


logger = logging.getLogger(__name__)


# 用户问题进入执行链路之前的预处理。
#
# 指代消解与意图识别原本写在 Tool Agent 和 PlannerAgent 各自的入口里。
# 引入统一入口 /ask 之后这段序列出现了第三处调用，因此抽到这里：
# 三条入口共用同一份实现，统一入口也只需要预处理一次。


TOOL_AGENT = "tool_agent"
PLANNER_AGENT = "planner_agent"

# 走 PlannerAgent 的意图。
#
# 判据是任务结构：组会汇报提纲和文献对比的步骤相对固定，
# 适合先出计划再按计划执行，换来稳定和可审查。
#
# 其余意图一律交给 Tool Agent，因为它的工具集更全——22 个，
# 包含 list_missing_files、list_literature_by_reader 这类结构化工具，
# 而 PlannerAgent 只有 5 个检索工具。
PLANNER_INTENTS = {
    "group_report",
    "compare_papers",
}


@dataclass
class PreparedQuestion:
    """预处理结果，可直接交给任一条执行链路。"""

    thread_id: str
    original_question: str
    rewrite: RewriteResult
    intent: IntentResult
    config: RunnableConfig
    settings: Settings


def route_for_intent(intent: str) -> str:
    """把意图映射到执行链路，返回 TOOL_AGENT 或 PLANNER_AGENT。

    默认走 Tool Agent。只有明确列在 PLANNER_INTENTS 里的才走 PlannerAgent。
    """

    if intent in PLANNER_INTENTS:
        return PLANNER_AGENT
    return TOOL_AGENT


def _stage_config(config: RunnableConfig | None, stage: str) -> RunnableConfig:
    """派生一份带阶段标记的 config。

    同一次请求里会有改写、意图识别、执行等多个模型调用。它们共用同一个
    session，靠 run_name 和 metadata.stage 在 Langfuse 里区分。
    """

    config = config or {}
    return {
        **config,
        "run_name": f"feishu-{stage.replace('_', '-')}",
        "metadata": {
            **config.get("metadata", {}),
            "stage": stage,
        },
    }


# 改写与意图合并调用用的系统提示词。由 coreference.REWRITE_SYSTEM_PROMPT
# 与 intent.INTENT_SYSTEM_PROMPT 融合而来：规则相同，输出合并成一个 JSON。
# 两处原提示词改动时需同步这里（意图合法性最终由 INTENT_WHITELIST 校验
# 兜底，改写长度由 MAX_REWRITE_* 校验兜底，提示词漂移不会放行坏结果）。
UNDERSTAND_SYSTEM_PROMPT = """你是一个飞书知识库 Agent 的查询理解模块，一次完成两件事：先把用户的最新问题改写，再判断改写后问题的任务类型。

── 任务一：查询改写 ──
把最新问题改写成不依赖对话历史也能独立理解的完整问题。
1. 如果最新问题已经能独立理解（没有代词、没有省略），rewritten_query 原样返回该问题。
2. 如果存在指代或省略，用对话历史里的信息补全。补全时优先用具体编号表达被指代的对象，
   例如把「它」写成「第001篇文献」。
3. 只使用历史里出现过的信息，不要编造。
4. 如果历史里找不到指代对象，rewritten_query 原样返回最新问题，item_id 留空。
5. 增量/比较类指令要扩写成完整任务：如果历史里正在做某项任务
   （如对比第003篇和第022篇的研究方法），而最新问题是往里加东西
   （如「加上第8篇」），rewritten_query 要写成本来的任务加上新对象
   （如「对比第003篇、第022篇和第008篇的研究方法」），
   不能只写「加上第8篇」这种离开历史就看不懂的半句话。

── 任务二：意图识别 ──
判断改写后的问题属于哪一类任务，只能选一个：
1. simple_qa：通用知识库问答。包括制度、流程、产品、项目、业务事实，以及不需要固定结构的普通文档问题。
2. metadata_query：元数据或统计查询。问文献编号、整理者、DOI、附件、PDF/PPT 缺失情况、文献清单、数量统计。
3. summary_paper：总结某篇论文。要论文的研究问题、方法、数据、结论、贡献。
4. summary_ppt：总结某篇 PPT 或汇报内容。
5. group_report：生成组会汇报提纲。需要按汇报顺序组织。
6. compare_papers：对比多篇文献。要相同点、不同点、适用场景。

意图判断规则：
- 无法判断时返回 simple_qa。
- 出现「有哪些」「哪些文献」「缺 PDF」「缺 PPT」「谁整理的」「统计」这类问法，优先考虑 metadata_query。
- 对比两位整理者的阅读清单（「某某和某某看的论文有什么不同」「谁的阅读方向偏什么」）是清单/元数据问题，选 metadata_query。
- 只有对比具体论文的研究内容（方法、数据、结论的异同）才选 compare_papers。
- 出现「提纲」「组会」「汇报」这类问法，考虑 group_report。
- 追问延续：如果当前问题是上一轮任务的增量指令（例如上一轮在对比 012
  和 045，现在说「再加上013」「把 013 也比进去」），应输出上一轮的任务类型。

先完成任务一，再基于改写得到的 rewritten_query 完成任务二。
只输出一个 JSON，不要输出解释或其他文字：
{
  "rewritten_query": "改写后的完整问题",
  "item_id": "三位编号，例如 001；没有就留空字符串",
  "intent": "simple_qa | metadata_query | summary_paper | summary_ppt | group_report | compare_papers",
  "reason": "改写与意图的判断依据"
}
"""


def _rewrite_fallback(question: str, reason: str) -> RewriteResult:
    """构造一个「不改写」的结果，降级语义与 coreference._fallback 一致。"""

    return RewriteResult(
        original_question=question,
        rewritten_query=question,
        item_id="",
        rewritten=False,
        reason=reason,
    )


def _intent_fallback(question: str, reason: str) -> IntentResult:
    """构造一个默认意图的结果，降级语义与 intent._fallback 一致。"""

    return IntentResult(question=question, intent=DEFAULT_INTENT, reason=reason)


def parse_understand_json(
    text: str, question: str
) -> tuple[RewriteResult, IntentResult]:
    """解析合并调用的输出。两个字段各自校验、各自降级，互不拖垮。

    改写侧校验与 coreference.parse_rewrite_json 相同（非空 + 长度上限），
    意图侧与 intent.parse_intent_json 相同（白名单）。reason 两边共用。
    """

    data = parse_llm_json(text)
    if not isinstance(data, dict):
        reason = "模型输出不是合法 JSON，改写回退原问题、意图回退 simple_qa。"
        return _rewrite_fallback(question, reason), _intent_fallback(question, reason)

    raw_query = data.get("rewritten_query")
    if isinstance(raw_query, str) and raw_query.strip():
        rewritten_query = raw_query.strip()
        if len(rewritten_query) > len(question) * MAX_REWRITE_RATIO + MAX_REWRITE_EXTRA:
            rewritten_query = question
            shared_reason = "模型改写输出长度异常，回退到原问题。"
        else:
            shared_reason = str(data.get("reason") or "").strip() or "模型完成改写。"
    else:
        rewritten_query = question
        shared_reason = "模型没有给出改写结果，回退到原问题。"

    rewrite = RewriteResult(
        original_question=question,
        rewritten_query=rewritten_query,
        item_id=normalize_item_id(data.get("item_id", "")),
        rewritten=rewritten_query != question,
        reason=shared_reason,
    )

    raw_intent = data.get("intent")
    if isinstance(raw_intent, str) and raw_intent.strip() in INTENT_WHITELIST:
        intent = IntentResult(
            question=rewritten_query,
            intent=raw_intent.strip(),
            reason=str(data.get("reason") or "").strip() or "模型完成意图识别。",
        )
    else:
        intent = _intent_fallback(
            rewritten_query,
            f"模型意图 {raw_intent!r} 缺失或不在允许列表，回退 simple_qa。",
        )

    return rewrite, intent


def understand_question(
    question: str,
    history: list,
    settings: Settings | None = None,
    config: RunnableConfig | None = None,
) -> tuple[RewriteResult, IntentResult]:
    """一次模型调用同时完成查询改写与意图识别。

    需要改写的追问场景原先要串行走两次 LLM（改写 → 意图），合并后省一次
    网络往返；首轮、带编号等无需改写的场景改写本来就不花调用，这里仍单独
    发意图调用，不比原来多花。意图基于改写结果的依赖关系保留在提示词内部
    （先改写、再据改写结果分类），不因合并而丢失。任何失败都不向上抛出，
    降级语义与两个原模块各自一致：改写回退原问题、意图回退 simple_qa。
    """

    settings = settings or get_settings()

    if not needs_rewrite(question, history):
        if not history:
            rewrite = _rewrite_fallback(question, "历史为空，首轮无需改写。")
        else:
            rewrite = _rewrite_fallback(question, "问题已含明确编号，无需改写。")
        intent = classify_intent(
            question,
            settings=settings,
            config=_stage_config(config, "intent"),
            history=history,
        )
        return rewrite, intent

    try:
        messages = [
            SystemMessage(content=UNDERSTAND_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"对话历史：\n{build_history_text(history)}\n\n"
                    f"最新问题：{question}"
                )
            ),
        ]
        response = build_llm(settings).invoke(
            messages, config=_stage_config(config, "query_understand")
        )
        return parse_understand_json(response.content, question)
    except Exception as exc:
        logger.warning("改写+意图合并调用失败，分别降级：%s", exc)
        reason = f"理解调用异常，改写回退原问题、意图回退 simple_qa：{exc}"
        return _rewrite_fallback(question, reason), _intent_fallback(question, reason)


def prepare_question(
    question: str,
    thread_id: str,
    mode: str,
    settings: Settings | None = None,
) -> PreparedQuestion:
    """执行指代消解与意图识别，并组装后续链路要用的 config。

    mode 决定 Langfuse 的 run_name 与 tags。统一入口传 "auto"，
    两条专用入口分别传自己的模式名。
    """

    settings = settings or get_settings()
    config = build_run_config(thread_id=thread_id, mode=mode)

    history = load_history(thread_id, settings=settings)
    # 改写与意图合并为一次模型调用（见 understand_question）：
    # 需要改写的追问省一次网络往返，无需改写的场景行为与原来一致。
    rewrite, intent = understand_question(
        question,
        history,
        settings=settings,
        config=config,
    )

    # 追问延续兜底：意图识别带着历史仍判成 simple_qa，而上一轮是
    # 任务型意图、追问里又带文献编号（「再加上013」）时，强制沿用
    # 上一轮任务意图。带编号是增量的强信号——不带编号的泛泛追问
    # （「为什么呢」）信任意图识别，避免把无关新问题劫持进任务链路。
    # 上一轮本身就是 simple_qa / 无历史时不触发。
    if (
        intent.intent == "simple_qa"
        and history
        and ITEM_ID_PATTERN.search(question)
    ):
        last = history[-1]
        prev_intent = getattr(last, "intent", "") or ""
        if prev_intent in SUMMARY_INTENTS:
            intent = IntentResult(
                question=intent.question,
                intent=prev_intent,
                reason=(
                    f"追问型问题且上一轮意图是 {prev_intent}，"
                    "沿用上一轮任务意图（原始判定 simple_qa）"
                ),
            )

    # 意图写进 ContextVar，本次请求内的工具调用（同一线程同一调用栈）
    # 可以读到——rerank 门槛按意图分档要用（见 app/query_context.py）。
    set_current_intent(intent.intent)

    # 意图补挂进跟踪标签：观测页的「意图成本」卡在 Langfuse 里按
    # intent:* 标签聚合 token（见 app/observability.attach_intent_tag）。
    attach_intent_tag(config, intent.intent)

    # 引用编号注册表同线程重置：本轮所有工具结果的「资料N」共用一套
    # 全局编号，答案里的 [n] 和前端来源面板靠它对齐（见 app/citations.py）。
    reset_citation_registry()

    return PreparedQuestion(
        thread_id=thread_id,
        original_question=question,
        rewrite=rewrite,
        intent=intent,
        config=config,
        settings=settings,
    )
