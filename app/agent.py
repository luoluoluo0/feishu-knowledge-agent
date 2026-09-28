import logging
import threading
from functools import lru_cache
from pathlib import Path

from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from app.config import Settings, get_settings
from app.chart_expert import ask_chart_expert
from app.context import build_context_hook
from app.conversation_store import append_turn
from app.coreference import Turn
from app.llm import build_llm
from app.pipeline import PreparedQuestion, prepare_question
from app.token_meter import read_token_usage
from app.tools import PaperSearchTools


# SqliteSaver 是可选依赖（langgraph-checkpoint-sqlite），缺失时降级到
# InMemorySaver，所以不能在模块顶层硬 import。
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    SqliteSaver = None


# 这一层是真正的“工具调用 Agent”。
# 和 PlannerAgent 的区别：
# - PlannerAgent：先让模型输出计划，代码按计划一步步执行。
# - Tool Agent：把工具交给模型，模型自己决定什么时候调用哪个工具。


_CHECKPOINTER_CONTEXTS: list = []
_CHECKPOINTER_LOCK = threading.Lock()


AGENT_SYSTEM_PROMPT = """你是一个飞书知识库检索与问答 Agent。
你可以检索从飞书同步的 Docx、PDF，以及兼容保留的文献卡片和 PPT 资料，
根据知识库证据回答制度、产品、项目、业务和科研文献等问题。

总原则：
1. 只根据工具返回的资料回答，不要编造资料中没有的信息。
2. 先判断用户问的是通用知识库内容，还是文献元数据、具体文献、跨文献比较或组会汇报。
3. 如果对话历史里已经确定过某份文档、某篇文献或 item_id，后续追问默认沿用该对象。
4. 如果资料不足，要直接说明“资料中没有充分说明”，不要硬凑答案。
5. 回答时尽量说明依据来自飞书 Docx、PDF 正文、文献卡片还是 PPT。
6. 证据里的「【图表解读】」是图表内容的权威转述（VLM 看图后整理的类型、
   坐标轴、图中事实），问图里有什么/图中数值/分布结构时，直接引用它作答，
   不要因为它是图片就回答“没有文字描述”；解读里没有的细节才如实说缺失。
7. 引用标注：答案里用到某条资料的具体信息时，在对应句子或条目末尾标注来源编号。
   只写方括号加数字，如 [7]；不要写成 [资料7] 或（资料7）。
   编号就是工具结果里「资料7」的数字 7。综合多条资料可以连写，如 [1][3]，
   但一处最多连写 2~3 个最相关的编号，不要罗列一长串。
   一般性转述或没有资料依据的说明不要标编号。

工具选择规则：

一、通用飞书知识库问答，优先检索同步的 Docx/PDF 正文。
- 适用于制度、流程、产品说明、项目资料、业务知识和普通文档问答。
- 使用 search_paper；这是为兼容旧版本保留的工具名，实际检索 Milvus 中的
  Docx/PDF 正文，并不只检索论文。
- 用户没有指定资料类型时，也可以使用 search_all 兜底。

二、文献精确元数据问题，优先用结构化工具，不要优先用向量检索。
适合问题：
- 某某讲过哪些论文？
- 某某整理了哪些文献？
- 第 062 篇是谁整理的？
- 某篇文献的 DOI、PDF、PPT、附件、状态是什么？
- 哪些文献缺 PDF / 缺 PPT / 资料不齐？

优先工具：
- list_literature_by_reader：按整理者列文献。
- get_literature_by_item_id：按 item_id 查单篇文献元信息。
- list_missing_files：列资料不齐文献。
- list_literature_by_status：按 ready / missing_pdf / missing_ppt / note_only 查。

三、跨文献主题找资料，优先用语义文献列表工具。
适合问题：
- 生计韧性相关论文有哪些？
- 能源贫困相关文献有哪些？
- 哪些论文讨论了农村家庭风险、恢复能力、贫困预测？
- 帮我找某个方向相关论文。

优先工具：
- semantic_search_literature：先返回候选文献列表。
- 如果用户继续追问某篇，再转入 item_id 相关工具。

四、查 Docx/PDF 正文内容，用 search_paper。
适合问题：
- 某个研究方法是什么？
- 某类论文用了什么模型、数据、理论框架？
- 跨文献比较研究问题、方法、结论。
- 用户没有明确指定具体文献，但问的是正文内容。

工具：
- search_paper：检索 Docx/PDF 正文。内含关键词与语义两路召回和重排，
  不需要再区分「纯向量」和「混合」。

五、查 PPT 或组会汇报内容，用 hybrid_search_ppt。
适合问题：
- PPT 讲了什么？
- 组会汇报结构是什么？
- 第几页幻灯片讲什么？
- 帮我整理汇报提纲中的 PPT 内容依据。

工具：
- hybrid_search_ppt。

六、查某一篇具体文献内的信息，先确定 item_id。
如果用户明确说了 item_id、编号、第几篇，或者历史已经确定某篇：
- 查基础信息、整理者、DOI、附件：get_literature_by_item_id。
- 查这篇的正文或 PPT：search_by_item。

注意：
- search_by_item 在指定文献范围内做混合检索，一次覆盖 PDF 正文、PPT 和文献卡片。
- 如果用户问“它的研究方法是什么”，且历史里有明确文献，必须沿用历史中的 item_id。

七、全库检索兜底。
只有当你无法判断该查文献卡片、PDF 还是 PPT 时，才用：
- search_all。

八、组会汇报提纲。
如果用户要生成某篇文献的组会汇报提纲：
1. 先确定具体文献和 item_id。
2. 用 get_literature_by_item_id 查文献基础信息。
3. 用 search_by_item 查这篇的研究问题、方法、数据、结论。
4. 用 hybrid_search_ppt 查 PPT 的汇报结构和重点。
5. 最后按“背景-研究问题-数据方法-核心发现-汇报结构-可讨论问题”组织回答。

回答要求：
1. 不要直接堆工具原文，要整理成用户能读懂的答案。
2. 如果检索到多篇相近文献，先列候选，再说明主要依据哪篇。
3. 如果是列表问题，优先用编号、标题、整理者、附件状态清晰列出。
4. 如果是论文内容问题，回答中要包含依据来源，例如“依据 PDF 正文第几页”或“依据 PPT 第几页”。
5. 如果工具结果明显不足或不相关，要说明不足，并建议换关键词、指定 item_id，或补充资料。
"""

# 图表专家（微调模型）启用时追加到系统提示词的段落。
# 只在 CHART_EXPERT_ENABLED=true 时拼入——工具不存在时这段话只会
# 诱导模型调用一个不存在的工具。
CHART_EXPERT_PROMPT_SUFFIX = """

九、图表细节问题（颜色、方位、数值、构成、标注方式）。
当用户问某张图「图里/图中/图上」的细节，或检索结果里出现「【图表解读】」时：
- 优先调用 chart_expert_answer，它会返回基于图表解读的权威作答；
- 拿到结果后结合其他资料综合回答，图文冲突时以图表专家的结论为准；
- 工具返回「没有命中图表块」时，再用常规检索工具兜底。"""


def build_langchain_tools(
    settings: Settings | None = None,
    paper_tools: PaperSearchTools | None = None,
):
    """把项目里的 PaperSearchTools 包装成 LangGraph 可以识别的工具。

    PaperSearchTools 是我们自己写的业务工具类。
    @tool 包装之后，create_react_agent 才能把这些函数交给大模型调用。

    paper_tools 不传时：默认配置走进程级单例（见 app/resources.py，
    构造一次约 10 秒 / 379MB，没必要建多份）；显式传入自定义 settings
    则保持原行为独立构造，供脚本和测试使用。
    """

    if paper_tools is None:
        if settings is not None:
            paper_tools = PaperSearchTools(settings)
        else:
            from app.resources import get_retrieval_stack

            paper_tools = get_retrieval_stack()

    @tool
    def search_all(query: str) -> str:
        """全库检索：不确定应查飞书 Docx、PDF、文献卡片还是 PPT 时使用。"""

        return paper_tools.search_all(query)

    @tool
    def list_literature_by_reader(reader: str) -> str:
        """按阅读整理者精确列出文献：适合回答“某某讲了哪些论文、某某分享过哪些文献”。"""

        return paper_tools.list_literature_by_reader(reader)

    @tool
    def semantic_search_literature(query: str) -> str:
        """用向量库语义检索相关文献：适合回答任意主题、方向、概念有哪些文献。"""

        return paper_tools.semantic_search_literature(query)

    @tool
    def list_literature_by_status(status: str) -> str:
        """按资料状态列出文献：status 可用 ready、missing_pdf、missing_ppt、note_only。"""

        return paper_tools.list_literature_by_status(status)

    @tool
    def get_literature_by_item_id(item_id: str) -> str:
        """按文献编号精确获取元信息：适合查某一篇的标题、整理者、DOI、PDF、PPT、状态。"""

        return paper_tools.get_literature_by_item_id(item_id)

    @tool
    def list_missing_files() -> str:
        """列出所有资料不齐全的文献：适合查缺 PDF、缺 PPT、资料缺失情况。"""

        return paper_tools.list_missing_files()

    @tool
    def hybrid_search_literature_card(query: str) -> str:
        """混合检索文献卡片：适合查标题、整理者、DOI、附件、文献基础信息。"""

        return paper_tools.hybrid_search_literature_card(query)

    @tool
    def search_paper(query: str) -> str:
        """检索飞书 Docx/PDF 正文：适合制度、产品、项目、业务和科研内容问答。"""

        return paper_tools.search_paper(query)

    @tool
    def hybrid_search_ppt(query: str) -> str:
        """混合检索 PPT：适合查组会汇报结构、幻灯片内容、PPT 重点。"""

        return paper_tools.hybrid_search_ppt(query)

    @tool
    def search_by_item(query: str, item_id: str) -> str:
        """按文献编号检索：当用户明确指定第几篇文献时使用，item_id 示例为 001。"""

        return paper_tools.search_by_item(query=query, item_id=item_id)

    # 只暴露这一份清单。
    #
    # 原先 22 个，其中 12 个是重复的或指向旧库：
    # - hybrid_* 与 search_* 在 Milvus 这边早已是同一条路径，纯向量那版
    #   是改造前的遗留；
    # - search_item_paper / _ppt / _card 三兄弟被 search_by_item 覆盖；
    # - list_literature_by_theme 的注释自己写着「兼容旧调用」；
    # - get_project_stats 读的还是旧的 chunks.jsonl（新的是 chunks_v2）；
    # - get_missing_materials_summary / get_reader_workload_summary 与
    #   list_missing_files / list_literature_by_reader 重叠。
    #
    # 每多一个工具，每次调用模型都要多读一份定义（实测平均 89 token），
    # 而且选项越多越容易选错。22 个降到 10 个，固定开销省约 1000 token。
    tools = [
        search_all,
        list_literature_by_reader,
        semantic_search_literature,
        list_literature_by_status,
        get_literature_by_item_id,
        list_missing_files,
        hybrid_search_literature_card,
        search_paper,
        hybrid_search_ppt,
        search_by_item,
    ]

    # 第 11 个工具：图表专家（自部署微调模型）。总闸关闭时不注册，
    # 工具定义的 token 开销和误触发风险都不进主路径。
    settings = settings or get_settings()
    if settings.chart_expert_enabled:

        @tool
        def chart_expert_answer(query: str) -> str:
            """回答图表细节问题：图中颜色、方位、数值、构成、标注方式等。当用户问某张图「图里/图中/图上」的细节，或检索结果里出现【图表解读】时优先调用。"""

            return ask_chart_expert(query, paper_tools, settings)

        tools.append(chart_expert_answer)

    return tools


def _create_sqlite_checkpointer(db_path: str):
    """真正创建一个 SQLite checkpointer，登记进全局列表以便退出时关闭。"""

    candidate = SqliteSaver.from_conn_string(db_path)
    # 不同版本的 SqliteSaver.from_conn_string 可能直接返回 saver，
    # 也可能返回上下文管理器。这里两种都兼容。
    if hasattr(candidate, "__enter__"):
        checkpointer = candidate.__enter__()
        with _CHECKPOINTER_LOCK:
            _CHECKPOINTER_CONTEXTS.append(candidate)
    else:
        checkpointer = candidate

    if hasattr(checkpointer, "setup"):
        checkpointer.setup()

    return checkpointer


@lru_cache
def _shared_sqlite_checkpointer(db_path: str):
    """同一个库路径复用同一个连接。

    缓存 Agent 失败后被清掉重建时，checkpointer 不能跟着重建：SQLite
    连接是所有请求共享的，重建会把老连接关掉，在途请求直接撞上
    "cannot operate on a closed database"。连接跟随进程生命周期，
    进程退出时由 close_checkpointers 统一关闭。
    """

    return _create_sqlite_checkpointer(db_path)


def build_checkpointer(settings: Settings):
    """创建 Agent 记忆保存器。

    优先使用 SQLite Checkpointer，把 LangGraph 的多轮状态写入数据库。
    如果当前环境没有安装 langgraph-checkpoint-sqlite，则临时回退到 InMemorySaver，
    这样服务还能启动，但重启后不会保留对话记忆。
    """

    if SqliteSaver is None:
        print(
            "未安装 langgraph-checkpoint-sqlite，"
            "当前使用 InMemorySaver，服务重启后会丢失 Agent 记忆。"
        )
        return InMemorySaver()

    checkpoint_path = Path(settings.checkpoint_db_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    return _shared_sqlite_checkpointer(str(checkpoint_path))


def close_checkpointers() -> None:
    """关闭全部 SQLite checkpointer 连接。只在进程退出时调用。

    之前失败路径也会调它：缓存 Agent 被清掉时顺手关连接，想减少
    Windows 下数据库文件被占用的概率。但连接是所有在途请求共享的，
    中途关闭等于让并发请求集体报错，现在只留给 API 的 lifespan 用。
    """

    with _CHECKPOINTER_LOCK:
        while _CHECKPOINTER_CONTEXTS:
            context = _CHECKPOINTER_CONTEXTS.pop()
            try:
                context.__exit__(None, None, None)
            except Exception:
                pass


def build_agent(settings: Settings | None = None):
    """创建标准工具调用 Agent。

    checkpointer 负责保存多轮对话状态。
    thread_id 相同的时候，Agent 可以接着同一段会话继续聊。
    """

    settings = settings or get_settings()
    llm = build_llm(settings)
    tools = build_langchain_tools(settings)
    checkpointer = build_checkpointer(settings)

    # 图表专家启用时，系统提示词追加对应的工具选择规则段。
    prompt = AGENT_SYSTEM_PROMPT
    if settings.chart_expert_enabled:
        prompt += CHART_EXPERT_PROMPT_SUFFIX

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
        checkpointer=checkpointer,
        # 上下文管理钩子：窗口模式（阈值触发、压完定死）优先；
        # 只影响发给模型的内容，checkpointer 里的原始记录一条不动。
        pre_model_hook=build_context_hook(settings),
    )


logger = logging.getLogger(__name__)


def find_dangling_tail(messages) -> list:
    """检测消息尾部的悬空工具调用段，返回需要删除的消息。

    流式请求中途断连，checkpoint 会停在半截：AI 消息带着 tool_calls
    存进去了，对应的 ToolMessage 还没来得及写。这种尾段会让下一次
    同 thread 请求被 LangGraph 校验直接拒绝（INVALID_CHAT_HISTORY）。

    从尾部向前收集到安全边界为止——HumanMessage（新一轮提问）或
    不带 tool_calls 的 AI 消息（上一轮的最终回答）。收集到的尾段里
    只要有任何一个 tool_call 没有配对的 ToolMessage，整段按悬空
    处理：那一轮当作没发生过，删掉即可恢复合法历史。
    """

    tail = []
    for message in reversed(messages):
        kind = getattr(message, "type", "")
        if kind == "human":
            break
        if kind == "ai" and not getattr(message, "tool_calls", None):
            break
        tail.append(message)
    tail.reverse()

    if not tail:
        return []

    called_ids = set()
    answered_ids = set()
    for message in tail:
        for call in getattr(message, "tool_calls", None) or []:
            called_ids.add(str(call.get("id") or ""))
        if getattr(message, "type", "") == "tool":
            answered_ids.add(str(getattr(message, "tool_call_id", "") or ""))

    if called_ids - answered_ids:
        return tail
    return []


def heal_dangling_tool_history(agent, config) -> int:
    """入口处自愈：截掉 checkpoint 尾部的悬空工具调用段，返回删除条数。

    失败只记日志不抛——自愈是尽力而为，不能反过来挡住本次请求。
    """

    try:
        state = agent.get_state(config)
        messages = (state.values or {}).get("messages", [])
        dangling = find_dangling_tail(messages)
        if not dangling:
            return 0
        removals = [
            RemoveMessage(id=message.id)
            for message in dangling
            if getattr(message, "id", None)
        ]
        if removals:
            agent.update_state(config, {"messages": removals})
            logger.warning(
                "检测到断连留下的悬空工具调用历史 %s 条，已自动截断。",
                len(removals),
            )
        return len(removals)
    except Exception as exc:
        logger.warning("悬空历史检测失败，跳过自愈：%s", exc)
        return 0


def run_agent(
    agent,
    question: str,
    thread_id: str = "feishu-agent-demo",
    prepared: PreparedQuestion | None = None,
) -> dict:
    """运行一次 Agent。

    输入给 create_react_agent 的固定格式是：
    {"messages": [HumanMessage(content=question)]}

    config 里的 thread_id 用来区分不同会话。

    检索用的是指代消解之后的查询，原问题写回会话历史。

    prepared 用于统一入口已经完成预处理的情况，避免重复改写与分类。
    不传则在本地预处理。
    """

    prepared = prepared or prepare_question(question, thread_id, mode="tool-agent")

    # 先自愈断连留下的悬空工具调用（再做引用编号续号，读到干净历史）
    heal_dangling_tool_history(agent, prepared.config)

    # 按会话历史续接引用编号（新编号接在上一回合最大「资料N」之后），
    # 否则多轮对话里新旧两套「资料1..N」并存，答案里的 [n] 指代不明。
    from app.citations import seed_registry_from_history

    seed_registry_from_history(agent, prepared.config)

    result = agent.invoke(
        {"messages": [HumanMessage(content=prepared.rewrite.rewritten_query)]},
        config=prepared.config,
    )

    messages = result.get("messages", [])
    answer = messages[-1].content if messages else ""

    append_turn(
        thread_id,
        Turn(
            question=question,
            rewritten_query=prepared.rewrite.rewritten_query,
            item_id=prepared.rewrite.item_id,
            answer=answer,
            intent=prepared.intent.intent,
        ),
        settings=prepared.settings,
    )

    # 本次请求全部 LLM 调用的 token（意图识别 + 改写 + Agent 循环），
    # 由挂在 config 上的 TokenMeter 累计（见 app/token_meter.py）。
    return {
        **result,
        "rewrite": prepared.rewrite,
        "intent": prepared.intent,
        "token_usage": read_token_usage(prepared.config),
    }
