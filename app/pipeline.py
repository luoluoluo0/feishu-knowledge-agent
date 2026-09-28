from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig

from app.config import Settings, get_settings
from app.citations import reset_citation_registry
from app.conversation_store import load_history
from app.coreference import ITEM_ID_PATTERN, SUMMARY_INTENTS, RewriteResult, rewrite_query
from app.intent import IntentResult, classify_intent
from app.observability import attach_intent_tag, build_run_config
from app.query_context import set_current_intent


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


def _stage_config(config: RunnableConfig, stage: str) -> RunnableConfig:
    """派生一份带阶段标记的 config。

    同一次请求里会有改写、意图识别、执行等多个模型调用。它们共用同一个
    session，靠 run_name 和 metadata.stage 在 Langfuse 里区分。
    """

    return {
        **config,
        "run_name": f"feishu-{stage.replace('_', '-')}",
        "metadata": {
            **config.get("metadata", {}),
            "stage": stage,
        },
    }


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
    rewrite = rewrite_query(
        question,
        history,
        settings=settings,
        config=_stage_config(config, "query_rewrite"),
    )

    # 对改写后的查询做分类；带上会话历史——追问是不是上一轮任务的
    # 增量（「再加上013」），只有看着历史才分得清。
    intent = classify_intent(
        rewrite.rewritten_query,
        settings=settings,
        config=_stage_config(config, "intent"),
        history=history,
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
