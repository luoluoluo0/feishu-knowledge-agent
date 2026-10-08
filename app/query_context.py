import contextvars

from app.config import Settings


# 把「当前请求的意图」从预处理层传到工具层。
#
# 意图在 prepare_question（pipeline 层）就确定了，但 rerank 门槛的消费点
# 在 PaperSearchTools._paper_retrieve（tools 层），中间隔着 LLM 自主选工具
# 的过程，没法靠函数参数传递。工具执行和 prepare_question 在同一线程的
# 同一调用栈上（同步 /ask、流式 worker 线程、PlannerAgent 内部三条路径
# 都如此），所以用 ContextVar 带过去。
#
# ContextVar 默认空字符串：不经过 prepare_question 的路径（CLI 脚本、
# 单测直接调工具）读到空值，回落到全局默认门槛，行为与分档功能上线前
# 一致。每个请求都会重写这个变量，线程池线程复用不会串。

_current_intent = contextvars.ContextVar("current_intent", default="")


def set_current_intent(intent: str) -> None:
    """记录当前请求的意图，供本次请求内的工具调用读取。"""

    _current_intent.set(intent or "")


def get_current_intent() -> str:
    """读取当前请求的意图；不在请求链路里时返回空字符串。"""

    return _current_intent.get()


# 当前登录用户的传递：与意图同款机制。写入点在 API 层
# （require_current_user 依赖 / SSE worker 首行——线程不继承 ContextVar，
# 跨线程必须显式重设，见 planner_agent.run_steps 的教训），消费点在
# 工具层（审计日志、将来的按用户隔离）。
_current_user = contextvars.ContextVar("current_user", default="")


def set_current_user(user_id: int | str) -> None:
    """记录当前请求的用户标识，供本次请求内的工具调用读取。"""

    _current_user.set(str(user_id or ""))


def get_current_user() -> str:
    """读取当前用户标识；不在请求链路里时返回空字符串。"""

    return _current_user.get()


# 总结类意图：查询是宽泛的（"总结这篇论文""对比两篇""出个提纲"），
# reranker 对这类查询和具体段落的相关性打分天然偏低（实测语料内
# 也只有 0.005~0.76），用事实类的 0.8 门槛会系统性误拒。
# 事实类（simple_qa / metadata_query）维持原门槛，拒答防线不动。
SUMMARY_INTENTS = {
    "summary_paper",
    "summary_ppt",
    "group_report",
    "compare_papers",
}


def min_score_for_intent(intent: str, settings: Settings) -> float:
    """按意图算出本次检索该用的 rerank 门槛。"""

    if not settings.rerank_threshold_by_intent:
        return settings.rerank_min_score
    if intent and intent in SUMMARY_INTENTS:
        return settings.rerank_min_score_summary
    return settings.rerank_min_score
