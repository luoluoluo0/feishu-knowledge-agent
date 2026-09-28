import logging
from dataclasses import dataclass, field
from typing import Any, Iterator

from langchain_core.messages import HumanMessage


# 把 Agent 的执行过程转成可以流式推送的事件。
#
# 为什么不让 Agent 整个走异步：它的 checkpointer 是 SqliteSaver，异步方法
# 直接抛 NotImplementedError——
#
#   The SqliteSaver does not support async methods.
#   Consider using AsyncSqliteSaver instead.
#
# 换成 AsyncSqliteSaver 要改 checkpointer 的生命周期管理（它的 from_conn_string
# 是异步上下文管理器，得在 lifespan 里一直开着）。而多轮对话的记忆正是靠它，
# 已经验证可用，为了流式去动它不划算。
#
# 所以这里走同步流式：agent.stream(stream_mode=["messages", "updates"])。
# 它拿到的东西和 astream_events 一样——token 级输出、工具调用事件——只是
# 跑在线程里，由 API 层用队列桥接到异步生成器。
#
# 事件类型：
#   stage  —— 预处理阶段（指代消解、意图识别）
#   tool   —— 正在调用某个工具
#   token  —— 最终答案的一个片段
#   done   —— 结束，带完整答案与耗时
#   error  —— 出错


logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {"type": self.type, **self.data}


def _tool_calls_in(payload: Any) -> list[dict]:
    """从 updates 事件的载荷里挖出工具调用。

    updates 给的是「某个节点执行完之后的状态」，结构随节点类型变化，
    所以这里不假设形状，只认消息对象上的 tool_calls 字段。
    """

    if not isinstance(payload, dict):
        return []

    messages = payload.get("messages")
    if messages is None:
        return []
    if not isinstance(messages, list):
        messages = [messages]

    calls: list[dict] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if isinstance(call, dict):
                calls.append(call)
    return calls


def iter_agent_events(
    agent,
    message: HumanMessage,
    config: dict,
) -> Iterator[StreamEvent]:
    """跑一次 Agent，把过程逐个事件吐出来。

    这是同步生成器——调用方负责放到线程里跑。
    """

    seen_calls: set[str] = set()

    for item in agent.stream(
        {"messages": [message]},
        config=config,
        stream_mode=["messages", "updates"],
    ):
        # 多模式流返回的是 (模式名, 数据) 元组。
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        mode, data = item

        if mode == "updates":
            for call in _tool_calls_in(data.get("agent") if isinstance(data, dict) else None):
                name = str(call.get("name") or "")
                # 按调用 id 去重而不是工具名：同名工具的第二次调用是独立的
                # 真实调用，按名去重会吞掉它的 tool 事件——api 层就漏判一个
                # 回合边界，中间回合的过渡文本会混进最终答案。
                dedup_key = str(call.get("id") or "") or name
                if not name or dedup_key in seen_calls:
                    continue
                seen_calls.add(dedup_key)
                args = call.get("args") or {}
                query = ""
                if isinstance(args, dict):
                    query = str(args.get("query") or args.get("theme") or "")
                yield StreamEvent("tool", {"name": name, "query": query})

            # 工具节点执行完的 updates 里带 ToolMessage（检索原文）。
            # 载荷形状是 {"tools": {"messages": [ToolMessage]}}——字典套消息
            # 列表，不是裸列表；取错一层会静默产出零事件（实测踩过）。
            tools_payload = data.get("tools") if isinstance(data, dict) else None
            tool_messages = []
            if isinstance(tools_payload, dict):
                tool_messages = tools_payload.get("messages") or []
            elif isinstance(tools_payload, list):
                tool_messages = tools_payload

            for message in tool_messages:
                if getattr(message, "type", "") != "tool":
                    continue
                text = str(getattr(message, "content", "") or "")
                if text:
                    yield StreamEvent(
                        "tool_result",
                        {
                            "name": str(getattr(message, "name", "") or "tool"),
                            "text": text,
                        },
                    )
            continue

        if mode != "messages":
            continue

        # messages 模式给的是 (消息片段, 元数据)。
        chunk, meta = data
        node = (meta or {}).get("langgraph_node")

        # 只有 agent 节点的输出才是答案。
        #
        # tools 节点也会吐 token，那是工具内部的模型调用（比如查询翻译），
        # 还有工具返回的原文。把它们当答案发给用户就穿帮了。
        if node != "agent":
            continue

        # 带 tool_call_chunks 的是模型在组织工具调用参数，不是答案。
        # 实测这些片段按字到达（args="信息"、args="分化"），
        # 直接发出去用户会看到半截的检索参数。
        if getattr(chunk, "tool_call_chunks", None):
            continue

        text = str(getattr(chunk, "content", "") or "")
        if text:
            yield StreamEvent("token", {"content": text})


def collect_answer(events: Iterator[StreamEvent]) -> tuple[str, list[dict]]:
    """把事件里的 token 拼回完整答案，同时收集工具调用。

    流式结束后要拿完整答案写会话历史，但那时候生成器已经跑完了，
    所以调用方要边转发边累积。这个函数只是给不需要转发时用。
    """

    parts: list[str] = []
    tools: list[dict] = []
    for event in events:
        if event.type == "token":
            parts.append(str(event.data.get("content", "")))
        elif event.type == "tool":
            tools.append(event.data)
            # 与 api 层同规则：工具调用标志前一回合是过渡文本，清空。
            parts.clear()
    return "".join(parts), tools
