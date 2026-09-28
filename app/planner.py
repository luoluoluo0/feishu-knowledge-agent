from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.config import Settings, get_settings
from app.conversation_store import Turn
from app.coreference import ITEM_ID_PATTERN
from app.json_utils import parse_llm_json
from app.llm import build_llm


# Planner 负责“先想怎么查”，不直接回答用户问题。
# 它和 Router 的区别：
# - Router：只选一条路线，比如只查 PDF 或只查 PPT。
# - Planner：可以安排多步，比如先查文献卡片，再查论文正文，再查 PPT。
ToolName = Literal[
    "search_all",
    "search_paper",
    "hybrid_search_ppt",
    "hybrid_search_literature_card",
    "search_by_item",
]


@dataclass
class PlanStep:
    """一次计划里的一个检索步骤。"""

    # 要调用哪个工具，比如 search_paper、hybrid_search_ppt。
    tool: ToolName

    # 传给工具的检索问题，不一定和用户原问题完全一样。
    query: str

    # 如果用户明确说了第几篇文献，可以放编号，比如 "001"。
    item_id: str = ""

    # 记录为什么要执行这一步，主要方便我们调试和学习。
    reason: str = ""


@dataclass
class PlanResult:
    """Planner 最终返回的结构化计划。"""

    # 任务类型，比如组会汇报、总结论文、总结 PPT。
    task_type: str

    # 具体要执行的步骤列表。
    steps: list[PlanStep] = field(default_factory=list)

    # 整体规划理由。
    reason: str = ""


PLANNER_SYSTEM_PROMPT = """你是一个飞书知识库 Agent 的 Planner。
你的任务不是直接回答问题，而是根据用户问题规划应该调用哪些检索工具。

可用工具：
1. hybrid_search_literature_card
   查文献卡片，适合获取标题、阅读整理者、DOI、附件、文献目录、基础背景信息。
2. search_paper
   查飞书同步的 Docx/PDF 正文，适合制度、流程、产品、项目、业务和科研内容。
   工具名为兼容旧版本保留，实际并不只检索论文。
3. hybrid_search_ppt
   查 PPT 内容，适合获取组会汇报结构、幻灯片标题、汇报重点、讲解顺序。
4. search_by_item
   查指定编号文献，适合用户明确说第001篇、第002篇等。
5. search_all
   全库检索，适合资料类型或范围不明确的问题。

任务类型 task_type：
- simple_qa：普通问答
- summary_paper：总结论文
- summary_ppt：总结 PPT
- group_report：生成组会汇报提纲
- compare_papers：对比文献

规划原则：
- 简单问题尽量 1 个 step。
- 复杂任务可以 2 到 4 个 step。
- 通用制度、流程、产品、项目或业务问题，优先使用 search_paper 检索 Docx/PDF 正文。
- 用户问题里出现文献编号时，正文检索步骤（search_paper、search_by_item）
  必须在 item_id 里带上该编号，query 只写检索正文用的内容问题，不要写编号。
- 对比多篇文献时，为每一篇各生成一个带 item_id 的 search_by_item 步骤，
  不要生成不带 item_id 的全库 search_paper 步骤。
- 生成组会汇报提纲时，必须优先同时包含 hybrid_search_literature_card、search_paper、hybrid_search_ppt。
- 如果用户明确说 PPT，必须包含 hybrid_search_ppt。
- 如果用户问阅读整理者、DOI、附件、文献基础信息，用 hybrid_search_literature_card。
- 如果用户问研究方法、结论、理论、数据、模型，用 search_paper。
- 如果用户问汇报结构、PPT讲了什么、组会提纲，用 hybrid_search_ppt。

请只输出 JSON，不要输出其他文字。
JSON 格式：
{
  "task_type": "simple_qa | summary_paper | summary_ppt | group_report | compare_papers",
  "reason": "为什么这样规划",
  "steps": [
    {
      "tool": "search_paper",
      "query": "检索问题",
      "item_id": "",
      "reason": "为什么调用这个工具"
    }
  ]
}
"""


ALLOWED_TOOLS = {
    "search_all",
    "search_paper",
    "hybrid_search_ppt",
    "hybrid_search_literature_card",
    "search_by_item",
}


def _normalize_item_id(item_id: str) -> str:
    """把用户或模型给出的编号统一成三位数，比如 1 -> 001。"""

    item_id = str(item_id or "").strip()
    if item_id.isdigit():
        return item_id.zfill(3)
    return item_id


def _has_tool(plan: PlanResult, tool_name: str) -> bool:
    """判断计划里是否已经包含某个工具。"""

    return any(step.tool == tool_name for step in plan.steps)


def _looks_like_group_report(question: str, task_type: str) -> bool:
    """判断这个问题是不是在要组会汇报提纲。"""

    keywords = ["组会", "汇报", "提纲", "分享", "报告"]
    return task_type == "group_report" or any(keyword in question for keyword in keywords)


def _question_item_id(question: str) -> str:
    """从问题里提取文献编号，比如「第 057 篇的组会提纲」-> 057。"""

    match = ITEM_ID_PATTERN.search(question)
    if not match:
        return ""
    return _normalize_item_id(match.group(1))


def _ensure_group_report_steps(plan: PlanResult, question: str) -> PlanResult:
    """给组会汇报类问题补齐必要步骤。

    大模型有时候会少规划一步，比如只查文献卡片和论文正文。
    但组会汇报通常还需要 PPT，因为 PPT 里最接近真实汇报顺序。
    所以这里做一层程序兜底，让计划更稳定。
    """

    if not _looks_like_group_report(question, plan.task_type):
        return plan

    plan.task_type = "group_report"
    query = question
    # 问题里点名了编号就带上下：限定范围后，总结型问题的低重排分
    # 不再触发「没找到资料」的误判（见 app/tools.py 的说明）。
    item_id = _question_item_id(question)

    if not _has_tool(plan, "hybrid_search_literature_card"):
        plan.steps.insert(
            0,
            PlanStep(
                tool="hybrid_search_literature_card",
                query=query,
                reason="组会汇报前先获取文献标题、整理者、附件等基础信息。",
            ),
        )

    if not _has_tool(plan, "search_paper"):
        plan.steps.append(
            PlanStep(
                tool="search_paper",
                query=f"{question} 研究问题 方法 结论 理论贡献",
                item_id=item_id,
                reason="组会汇报需要论文正文中的研究问题、方法、结论和贡献。",
            )
        )

    if not _has_tool(plan, "hybrid_search_ppt"):
        plan.steps.append(
            PlanStep(
                tool="hybrid_search_ppt",
                query=f"{question} PPT 汇报结构 幻灯片重点",
                reason="组会汇报提纲需要参考 PPT 的讲解结构和页面重点。",
            )
        )

    return plan


def parse_plan_json(text: str, fallback_question: str) -> PlanResult:
    """把大模型输出的 JSON 解析成 PlanResult。

    如果大模型没有输出合法 JSON，就退回成一次全库检索。
    这样 Agent 不会因为 Planner 输出格式不稳定就直接崩掉。
    """

    data = parse_llm_json(text)
    if not isinstance(data, dict):
        plan = PlanResult(
            task_type="simple_qa",
            reason="Planner 输出不是合法 JSON，回退到全库检索。",
            steps=[
                PlanStep(
                    tool="search_all",
                    query=fallback_question,
                    reason="兜底检索。",
                )
            ],
        )
        return _ensure_group_report_steps(plan, fallback_question)

    steps = []
    for raw_step in data.get("steps", []):
        tool = raw_step.get("tool", "search_all")
        if tool not in ALLOWED_TOOLS:
            tool = "search_all"

        steps.append(
            PlanStep(
                tool=tool,
                query=raw_step.get("query", "") or fallback_question,
                item_id=_normalize_item_id(raw_step.get("item_id", "")),
                reason=raw_step.get("reason", "") or "",
            )
        )

    if not steps:
        steps = [
            PlanStep(
                tool="search_all",
                query=fallback_question,
                reason="没有可执行步骤，回退到全库检索。",
            )
        ]

    # 问题里点名了编号、而模型生成的正文检索步骤没带 item_id 时，
    # 用问题里的编号补上。不带范围的全库检索对总结型问题必然过不了
    # 重排门槛，会被误判成「没找到资料」（见 app/tools.py 的说明）。
    question_item = _question_item_id(fallback_question)
    if question_item:
        for step in steps:
            if step.tool == "search_paper" and not step.item_id:
                step.item_id = question_item

    plan = PlanResult(
        task_type=data.get("task_type", "simple_qa") or "simple_qa",
        reason=data.get("reason", "") or "",
        steps=steps,
    )
    return _ensure_group_report_steps(plan, fallback_question)


class QueryPlanner:
    """根据用户问题生成多步检索计划。"""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.llm = build_llm(self.settings)

    def plan(
        self,
        question: str,
        config: RunnableConfig | None = None,
        history: list["Turn"] | None = None,
    ) -> PlanResult:
        """调用大模型生成计划，然后解析成 PlanResult。

        history 是会话历史（不含本轮）。追问经常只说增量——
        「再加上 013」「那 PPT 呢」——不带历史，规划器根本不知道
        要查哪些文献。指代消解会先改写一轮，但它是碰运气；
        这里把历史直接给规划器兜底。
        """

        context_text = question
        history_lines = [
            f"- 用户：{turn.question.strip()}"
            for turn in (history or [])
            if (turn.question or "").strip()
        ]
        if history_lines:
            context_text = (
                "对话历史（最近的在前，供理解当前问题；不要为历史单独规划步骤）：\n"
                + "\n".join(history_lines)
                + f"\n\n当前用户问题：{question}"
            )

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=context_text),
        ]
        response = self.llm.invoke(messages, config=config)
        return parse_plan_json(response.content, fallback_question=question)
