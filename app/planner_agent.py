from dataclasses import dataclass

from app.config import Settings, get_settings
from app.conversation_store import append_turn, load_history
from app.coreference import Turn
from app.llm import build_llm
from app.pipeline import PreparedQuestion, prepare_question
from app.planner import PlanResult, PlanStep, QueryPlanner
from app.token_meter import read_token_usage
from app.tools import PaperSearchTools


@dataclass
class PreparedRun:
    """一次 Planner 执行的准备产物：计划、检索结果与拼好的 prompt。

    prepare_execution 产出它，最终生成（invoke 或 stream）和收尾
    （record_turn）都吃它——流式与非流式共用同一段准备逻辑。
    """

    question: str
    prepared: PreparedQuestion
    history: list[Turn]
    plan: PlanResult
    step_results: list[dict]
    prompt: str


# PlannerAgent 是当前项目里最像“Agent”的部分。
# 它的完整流程是：
# 1. QueryPlanner 先根据用户问题生成检索计划。
# 2. PlannerAgent 按计划调用 PaperSearchTools。
# 3. 最后把多个工具结果交给 LLM，生成最终回答。


def format_plan(plan: PlanResult) -> str:
    """把计划格式化成文本，方便打印，也方便塞进最终 prompt。"""

    lines = [
        f"任务类型：{plan.task_type}",
        f"规划理由：{plan.reason}",
        "步骤：",
    ]

    for index, step in enumerate(plan.steps, start=1):
        lines.append(
            f"{index}. tool={step.tool}, item_id={step.item_id}, "
            f"query={step.query}, reason={step.reason}"
        )

    return "\n".join(lines)


# 拼进 prompt 的历史轮数与答案截断长度。
# Planner 的最终 prompt 本来就大（多篇论文的检索结果），历史只带
# 任务级上下文：问题原文 + 答案开头（提纲/对比的框架在前 800 字里）。
PLANNER_HISTORY_TURNS = 3
PLANNER_HISTORY_ANSWER_CHARS = 800


def format_history_block(history: list[Turn]) -> str:
    """把会话历史格式化成 prompt 里的一段。

    空 / 截断都发生在这一层，调用方只管拼。没有历史返回空串，
    prompt 里就不出现这一节。
    """

    usable = [turn for turn in history if (turn.question or "").strip()]
    if not usable:
        return ""

    lines = ["## 对话历史（按时间正序，供理解当前问题用）"]
    for turn in usable:
        answer_head = (turn.answer or "").strip()[:PLANNER_HISTORY_ANSWER_CHARS]
        lines.append(f"用户：{turn.question.strip()}")
        if answer_head:
            lines.append(f"助手：{answer_head}{'…' if len(turn.answer or '') > PLANNER_HISTORY_ANSWER_CHARS else ''}")
    lines.append("（历史结束。当前问题可能引用历史里的文献编号或内容，请结合理解。）")
    return "\n".join(lines)


class PlannerAgent:
    """可控 Planner Agent。

    它比普通 RAG 多了“规划”这一步：
    先决定查什么，再执行检索，最后综合回答。
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.planner = QueryPlanner(self.settings)
        # 默认配置走进程级共享检索栈（构造约 10 秒 / 379MB，没必要
        # 和 Tool Agent 各建一份）；显式传入自定义配置时独立构造。
        if settings is not None:
            self.tools = PaperSearchTools(self.settings)
        else:
            from app.resources import get_retrieval_stack

            self.tools = get_retrieval_stack()
        self.llm = build_llm(self.settings)

    def run_step(self, step: PlanStep) -> str:
        """执行计划中的一步。

        step.tool 决定调用哪个工具。
        step.query 是这个工具真正拿去检索的文本。
        """

        if step.tool == "hybrid_search_literature_card":
            return self.tools.hybrid_search_literature_card(step.query)

        if step.tool == "search_paper":
            # 计划里带 item_id 时限定到那一篇。之前这里把 item_id 直接
            # 丢掉了：组会提纲这类总结型问题在全库范围做重排，分数必然
            # 过不了门槛，被误判成「没找到资料」。
            return self.tools.search_paper(step.query, item_id=step.item_id or None)

        if step.tool == "hybrid_search_ppt":
            return self.tools.hybrid_search_ppt(step.query)

        if step.tool == "search_by_item" and step.item_id:
            return self.tools.search_by_item(step.query, item_id=step.item_id)

        return self.tools.search_all(step.query)

    def run_steps(self, plan: PlanResult, on_step=None) -> list[dict]:
        """按顺序执行所有计划步骤，并保存每一步的工具结果。

        on_step(step_result)：每完成一步回调一次——SSE 层用它把
        「正在检索第几步」实时推给前端，而不是全部跑完才露面。
        """

        results = []

        for index, step in enumerate(plan.steps, start=1):
            tool_result = self.run_step(step)
            step_result = {
                "step_index": index,
                "tool": step.tool,
                "query": step.query,
                "item_id": step.item_id,
                "reason": step.reason,
                "result": tool_result,
            }
            results.append(step_result)
            if on_step is not None:
                on_step(step_result)

        return results

    def build_final_prompt(
        self,
        question: str,
        plan: PlanResult,
        step_results: list[dict],
        history: list[Turn] | None = None,
    ) -> str:
        """把用户问题、计划、工具结果拼成最终回答 prompt。

        history 是会话历史（不含本轮）——「刚才提纲的第 3 点展开讲讲」
        这类追问必须看见上一轮答案才能接住，指代消解只改写问题，
        救不了这种。
        """

        result_texts = []
        for step_result in step_results:
            result_texts.append(
                f"步骤{step_result['step_index']}\n"
                f"工具：{step_result['tool']}\n"
                f"检索问题：{step_result['query']}\n"
                f"工具理由：{step_result['reason']}\n"
                f"检索结果：\n{step_result['result']}"
            )

        history_block = format_history_block(history or [])
        history_section = f"\n{history_block}\n" if history_block else ""

        return f"""你是一个飞书知识库检索与问答助手。
请只根据工具检索结果回答，不要编造资料中没有的信息。
如果资料不足，请明确说明不足在哪里。
引用标注：答案中用到某条资料的具体信息时，在对应句子或条目末尾标注来源编号。
只写方括号加数字，如 [7]；不要写成 [资料7] 或（资料7）。
编号就是检索结果里「资料7」的数字 7；综合多条资料可以连写 [1][3]，
但一处最多连写 2~3 个最相关的编号，不要罗列一长串。
一般性转述或没有资料依据的说明不要标编号。
{history_section}
用户问题：
{question}

执行计划：
{format_plan(plan)}

工具结果：
{chr(10).join(result_texts)}

请根据任务类型回答：
        - 如果是普通问答，直接回答并列出来自 Docx/PDF 等知识库资料的依据。
- 如果是总结论文，按“研究问题、方法、主要结论、意义、依据来源”组织。
- 如果是总结 PPT，按“PPT主题、主要内容、结构、依据来源”组织。
- 如果是组会汇报提纲，生成适合组会分享的分点提纲，尽量包含开场、研究背景、研究问题、方法、结果、贡献、不足、可讨论问题，并列出依据来源。
- 如果是对比文献，按相同点、不同点、适用场景、依据来源组织。
"""

    def prepare_execution(
        self,
        question: str,
        thread_id: str = "feishu-planner-demo",
        prepared: PreparedQuestion | None = None,
        on_step=None,
    ) -> "PreparedRun":
        """执行最终回答之前的全部步骤：预处理 → 计划 → 检索 → 拼 prompt。

        流式与非流式共用的准备阶段——这一段没有面向用户的 token，
        SSE 层通过 on_step 把每步检索实时推给前端。
        """

        prepared = prepared or prepare_question(
            question,
            thread_id,
            mode="planner-agent",
            settings=self.settings,
        )
        query = prepared.rewrite.rewritten_query

        # 会话历史：在 record_turn 写入之前读，拿到的只含之前的轮次。
        # 计划生成与最终回答共用——计划要知道上一轮在聊什么才能
        # 规划对文献，回答要看见上一轮产出才能接追问。
        history = load_history(thread_id, settings=self.settings)

        plan = self.planner.plan(query, config=prepared.config, history=history)
        step_results = self.run_steps(plan, on_step=on_step)
        prompt = self.build_final_prompt(query, plan, step_results, history=history)

        return PreparedRun(
            question=question,
            prepared=prepared,
            history=history,
            plan=plan,
            step_results=step_results,
            prompt=prompt,
        )

    def record_turn(self, run: "PreparedRun", answer: str) -> None:
        """把一轮问答写回会话历史。流式与非流式收尾共用。"""

        append_turn(
            run.prepared.thread_id,
            Turn(
                question=run.question,
                rewritten_query=run.prepared.rewrite.rewritten_query,
                item_id=run.prepared.rewrite.item_id,
                answer=answer,
                intent=run.prepared.intent.intent,
            ),
            settings=run.prepared.settings,
        )

    def answer(
        self,
        question: str,
        thread_id: str = "feishu-planner-demo",
        prepared: PreparedQuestion | None = None,
    ) -> dict:
        """完整执行一次 Planner Agent 问答（非流式）。

        检索用的是指代消解之后的查询，原问题写回会话历史。

        prepared 用于统一入口已经完成预处理的情况，避免重复改写与分类。
        不传则在本地预处理。
        """

        run = self.prepare_execution(question, thread_id, prepared)
        response = self.llm.invoke(run.prompt, config=run.prepared.config)
        self.record_turn(run, response.content)

        return {
            "question": run.question,
            "rewrite": run.prepared.rewrite,
            "intent": run.prepared.intent,
            "plan": run.plan,
            "step_results": run.step_results,
            "answer": response.content,
            # 计划 + 检索步骤 + 最终回答的全部 LLM 调用都在 prepared.config
            # 的 callbacks 上，TokenMeter 已累计（见 app/token_meter.py）。
            "token_usage": read_token_usage(run.prepared.config),
        }
