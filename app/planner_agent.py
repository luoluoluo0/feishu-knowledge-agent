from concurrent.futures import ThreadPoolExecutor, as_completed
import contextvars
from dataclasses import dataclass
import hashlib
import re

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


def _bind_output_cap(llm, max_output_tokens: int):
    """给 Planner 的模型实例绑定输出 token 上限。

    Planner 的最终回答是提纲/对比类长文，decode 时间与输出长度成正比，
    硬上限直接砍总耗时。bind 会同时作用于 answer() 的 invoke 和 api 层
    的 stream；0 表示不限制，原样返回。只影响 Planner，不碰 tool_agent
    侧的模型实例。
    """

    if max_output_tokens > 0:
        return llm.bind(max_tokens=max_output_tokens)
    return llm


# 拼装层瘦身的块边界：工具结果文本里每个资料块以「资料N」行开头。
_BLOCK_HEADER = re.compile(r"\n?(?=资料\d+\n)")


def _block_identity(block: str) -> str:
    """一个资料块的内容指纹：剥掉块号行、压平空白后取哈希。

    同一个 chunk 被多个步骤捞回时，工具产出的文本逐字相同、只有
    「资料N」行里的编号不同，剥掉编号行才能对上。同一页的两个不同
    块内容不同，哈希也不同——不会误伤。
    """

    body = re.sub(r"^资料\d+\n", "", block.strip())
    body = re.sub(r"\s+", "", body)
    return hashlib.md5(body.encode("utf-8")).hexdigest()


def _trim_step_result(result: str, top_k: int, seen_keys: set[str]) -> str:
    """拼装层瘦身：跨步去重 + 每步只保留重排头部 top_k 条。

    只剪「装订进最终 prompt 的页数」，不碰检索层——块已经在重排结果里
    排好序，取头部是复用已有的排序判断。去重按内容指纹，零信息损失。
    没有「资料N」块的文本（如「没有检索到」的降级说明）原样保留。
    """

    blocks = [b for b in re.split(_BLOCK_HEADER, result) if b.strip()]
    if not any(re.match(r"^资料\d+\n", b) for b in blocks):
        return result

    kept: list[str] = []
    kept_in_step = 0
    for block in blocks:
        if not re.match(r"^资料\d+\n", block):
            kept.append(block)
            continue
        identity = _block_identity(block)
        if identity in seen_keys:
            continue
        if top_k > 0 and kept_in_step >= top_k:
            continue
        seen_keys.add(identity)
        kept_in_step += 1
        kept.append(block)
    return "\n".join(kept)


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
        self.llm = _bind_output_cap(
            build_llm(self.settings), self.settings.planner_max_output_tokens
        )

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
        """并行执行所有计划步骤，并保存每一步的工具结果。

        计划在执行前一次性生成，步骤之间没有数据依赖——没有任何一步
        消费另一步的输出——所以可以并行检索（串行版在这里花掉
        步数×单步耗时）。两点必须保住：

        - 返回列表仍按步骤顺序排列，最终 prompt 里「步骤N」的编号
          不能乱；
        - on_step 按实际完成顺序触发，SSE 层的「正在检索第几步」
          更及时，但不再与步骤编号顺序一致（前端只展示进度，无影响）。

        底层检索栈与 tool_agent 的 langgraph 并行工具调用共用同一批
        实例（Milvus gRPC / httpx 均线程安全），不引入新的共享状态。
        """

        steps = list(enumerate(plan.steps, start=1))
        results: list[dict | None] = [None] * len(steps)

        def run_one(index: int, step: PlanStep) -> dict:
            return {
                "step_index": index,
                "tool": step.tool,
                "query": step.query,
                "item_id": step.item_id,
                "reason": step.reason,
                "result": self.run_step(step),
            }

        max_workers = min(self.settings.planner_max_parallel_steps, len(steps))
        if max_workers <= 1:
            for index, step in steps:
                step_result = run_one(index, step)
                results[index - 1] = step_result
                if on_step is not None:
                    on_step(step_result)
            return [r for r in results if r is not None]

        # 线程池新线程不会继承调用方的 ContextVar：意图（重排门槛、
        # 参考文献过滤都靠它）和「资料N」引用注册表都存在 ContextVar
        # 里，直接 submit 会在池线程里全部读到默认值——意图分档静默
        # 失效、每个线程各发各的资料编号。每个任务各自 copy_context()
        # 快照（提交时在调用方线程拍下）；注意一个 Context 对象同时
        # 只能被一个线程 enter，所有任务共用同一个会直接 RuntimeError。
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(contextvars.copy_context().run, run_one, index, step)
                for index, step in steps
            ]
            for future in as_completed(futures):
                step_result = future.result()
                results[step_result["step_index"] - 1] = step_result
                if on_step is not None:
                    on_step(step_result)
        return [r for r in results if r is not None]

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
        # 拼装层瘦身：去重 + 每步 top_k。top_k=0 表示不限制条数。
        seen_keys: set[str] = set()
        top_k = self.settings.planner_prompt_top_k_per_step
        for step_result in step_results:
            trimmed = _trim_step_result(
                str(step_result["result"]), top_k, seen_keys
            )
            result_texts.append(
                f"步骤{step_result['step_index']}\n"
                f"工具：{step_result['tool']}\n"
                f"检索问题：{step_result['query']}\n"
                f"工具理由：{step_result['reason']}\n"
                f"检索结果：\n{trimmed}"
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
长度约束：直接给出结论性内容，不要复述资料原文段落；提纲类每点一行、
先要点后依据；总篇幅控制在千字以内，把篇幅留给用户问的内容。
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
