"""精简每次发给模型的历史消息。

【默认关闭。为什么，见文件末尾。逻辑留着是因为改对之后还要用。】

上下文里八成是检索回来的论文原文（实测工具结果占 76~84%）。这些原文
在当轮是必需的——模型要靠它们组织答案；一旦变成历史，看起来就没必要
每轮重发了。

所以分两段处理：

- 本轮（最后一条用户消息之后）：一个字不动。本轮检索到的资料模型还要
  拿来回答，这时候动它就是自断粮草。
- 本轮之前：用户原话一个字不动，模型以前的回答留开头，检索原文换成
  一行提示。

关键约定：这里返回的是 llm_input_messages，它只作为本次调用模型的输入，
**不会覆盖 state 里的 messages**。也就是说 checkpointer 里的原始记录一条
不少，随时可以把这段逻辑摘掉，历史立刻恢复完整。

--------------------------------------------------------------------------

为什么默认关闭

拿 20 条 coreference 多轮用例做 A/B：

    关闭    18/20 = 90%     中位延迟 20920 ms
    开启    18/20 = 90%     中位延迟 40344 ms

准确率一点没掉，连失败的用例都完全相同。但慢了一倍。

原因是前缀缓存。DeepSeek 对相同前缀的 token 走缓存价，实测命中率约 67%：

    第 1 次调用：input 761，命中缓存 0
    第 2 次调用：input 761，命中缓存 512

全量重发的时候，第 N 轮的提示就是「第 N-1 轮的全部内容 + 新增」，
前缀天然稳定，缓存一直命中。而这里每轮都重拼历史，从第二轮起前缀就
变了，缓存全部失效。

于是省下来的是本来就命中缓存的 token——那些又便宜又快；真正需要重算
的 token 反而变多了。

要做这件事，方向是「压完定死」：压缩的边界一旦确定就不再回头重写，
两次压缩之间保持前缀不变。那是另一套设计。
"""

import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import Settings, get_settings


logger = logging.getLogger(__name__)

# 从工具返回的原文里抠标题。format_chunk 拼出来的每段都以「标题：xxx」开头。
_TITLE_PATTERN = re.compile(r"^标题：(.*)$", re.M)

# 提示里最多列几个标题、每个标题保留多少字。
MAX_TITLES_SHOWN = 5
TITLE_CHARS = 28

# 提不到标题时保留开头多少字。多用于结构化列表（文献清单、统计数字），
# 整条丢掉会让模型完全不知道那次查了什么。
FALLBACK_HEAD_CHARS = 100


def _text_of(message) -> str:
    """取消息的文本内容。多模态消息的 content 是列表，这里只拼 text 部分。"""

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _summarize_tool_message(message: ToolMessage) -> ToolMessage:
    """把检索原文换成一行提示。

    必须保留 tool_call_id——它是对上前面那条 tool_calls 的凭据，
    丢了接口会直接报错。
    """

    text = _text_of(message)
    titles = [title.strip() for title in _TITLE_PATTERN.findall(text) if title.strip()]

    if titles:
        shown = "、".join(title[:TITLE_CHARS] for title in titles[:MAX_TITLES_SHOWN])
        tail = f"等 {len(titles)} 篇" if len(titles) > MAX_TITLES_SHOWN else ""
        summary = (
            f"（此处原有 {len(titles)} 段检索结果，来自：{shown}{tail}。"
            "需要细节请重新调用检索工具。）"
        )
    else:
        summary = (
            f"{text[:FALLBACK_HEAD_CHARS].strip()}"
            f"……（后略，原文 {len(text)} 字。需要细节请重新调用工具。）"
        )

    extra = {}
    name = getattr(message, "name", None)
    if name:
        extra["name"] = name

    return ToolMessage(
        content=summary,
        tool_call_id=message.tool_call_id,
        **extra,
    )


def _slim_one(message, answer_head_chars: int):
    """精简一条历史消息。"""

    if isinstance(message, ToolMessage):
        return _summarize_tool_message(message)

    if isinstance(message, AIMessage):
        # 带工具调用的消息一个字都不能改：后面那条 ToolMessage 要靠它的
        # tool_calls 才能对上号，替换掉会让接口报错。
        if getattr(message, "tool_calls", None):
            return message

        text = _text_of(message)
        if len(text) <= answer_head_chars:
            return message
        return AIMessage(content=f"{text[:answer_head_chars]}……（后略）")

    # 用户原话原样保留。指代消解和模型理解都依赖它的精确措辞。
    return message


def _last_human_index(messages) -> int:
    """最后一条用户消息的位置。

    它就是「本轮」的起点。在它之后的消息（本轮检索到的资料、模型正在
    组织答案的过程）全部保真。
    """

    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return index
    # 没有用户消息（不该出现），全部当本轮处理，不做任何精简。
    return 0


def slim_messages(messages, *, answer_head_chars: int = 300):
    """把历史消息精简成给模型看的一版。

    入参和返回都是消息列表，不修改原列表。
    """

    if not messages:
        return messages

    start = _last_human_index(messages)
    return [
        message if index >= start else _slim_one(message, answer_head_chars)
        for index, message in enumerate(messages)
    ]


def build_slim_hook(settings: Settings | None = None):
    """构造挂在 Agent 上的精简钩子。

    关闭时返回 None——create_react_agent 的 pre_model_hook 默认就是 None，
    传进去等同于没装，不需要额外分支。
    """

    settings = settings or get_settings()
    if not settings.context_slim_enabled:
        return None

    head_chars = settings.context_answer_head_chars

    def slim_history(state):
        """每次调用模型之前跑一遍。"""

        messages = state.get("messages") or []
        slimmed = slim_messages(messages, answer_head_chars=head_chars)

        if logger.isEnabledFor(logging.DEBUG):
            before = sum(len(_text_of(message)) for message in messages)
            after = sum(len(_text_of(message)) for message in slimmed)
            logger.debug(
                "上下文精简：%d 条消息，%d 字 -> %d 字（省 %d%%）",
                len(messages),
                before,
                after,
                int((1 - after / before) * 100) if before else 0,
            )

        return {"llm_input_messages": slimmed}

    return slim_history


# --------------------------------------------------------------------------
# 上下文窗口：阈值触发的「压完定死」。
#
# 与上面被否决的每轮精简的关键区别：历史估算 token 没超阈值时**原样透传**
# （前缀稳定，DeepSeek 前缀缓存照常命中，零成本）；超过阈值才把最旧的
# 轮次摘要化，压完后历史变小，之后每轮继续透传——两次压缩之间前缀不变。
# 压缩是确定性的（同一输入永远得到同一输出），不引入任何随机性。

def estimate_tokens(messages) -> int:
    """粗估一段消息的 token 数。

    项目实测：中文约 1.3~1.9 字/词元，英文约 4 字符/词元。这里 CJK 字按
    0.8 词元、其他字符按 0.25 词元计——整体偏保守（高估），宁可提前压缩
    也不要贴着模型窗口爆掉。
    """

    total = 0
    for message in messages:
        for char in _text_of(message):
            total += 0.8 if "\u4e00" <= char <= "\u9fff" else 0.25
        # 每条消息的 role/结构开销。
        total += 4
    return int(total)


def _group_by_turn(messages) -> list[list]:
    """按「轮」分组：每条用户消息开启一组，其后所有消息归入同一组。

    整组处理保证 tool_calls 与 ToolMessage 永远同组——拆开会让接口报错。
    开头没有用户消息的散块（理论不该出现）单独成组，一并处理。
    """

    groups: list[list] = []
    for message in messages:
        if isinstance(message, HumanMessage) or not groups:
            groups.append([message])
        else:
            groups[-1].append(message)
    return groups


def window_messages(messages, *, max_tokens: int, answer_head_chars: int = 300):
    """把超窗的历史压缩成给模型看的一版。本轮（最后一条用户消息起）保真。

    策略：从最新往回累积整组原文，装不下的更旧轮次逐条摘要化（复用
    _slim_one：检索原文→一行提示、旧 AI 回答截头、用户原话不动）。
    摘要化的消息很小（约一两百字），所以边界附近自然形成「近期原文 +
    远期摘要」的分布。
    """

    if not messages:
        return messages

    current_start = _last_human_index(messages)
    current_turn = messages[current_start:]
    history = messages[:current_start]

    budget = max_tokens - estimate_tokens(current_turn)
    if budget <= 0:
        # 本轮自己就超了——历史全部摘要化，保真部分一个字不动。
        return [
            _slim_one(message, answer_head_chars) for message in history
        ] + list(current_turn)

    groups = _group_by_turn(history)
    kept_tokens = 0
    split = 0  # 前 split 组摘要化，其余原文保留
    for index in range(len(groups) - 1, -1, -1):
        group_tokens = estimate_tokens(groups[index])
        if kept_tokens + group_tokens > budget:
            split = index + 1
            break
        kept_tokens += group_tokens
    else:
        split = 0

    result = []
    for group in groups[:split]:
        result.extend(_slim_one(message, answer_head_chars) for message in group)
    result.extend(history[sum(len(g) for g in groups[:split]) :])
    result.extend(current_turn)
    return result


def build_window_hook(settings: Settings | None = None):
    """构造窗口钩子：阈值内透传（返回空 dict，即不改动模型输入）。"""

    settings = settings or get_settings()
    max_tokens = settings.context_window_max_tokens
    head_chars = settings.context_answer_head_chars

    def window_history(state):
        messages = state.get("messages") or []
        if estimate_tokens(messages) <= max_tokens:
            return {}

        trimmed = window_messages(
            messages, max_tokens=max_tokens, answer_head_chars=head_chars
        )
        logger.info(
            "上下文超窗（估算 %d token > %d），已压缩至 %d 条消息",
            estimate_tokens(messages),
            max_tokens,
            len(trimmed),
        )
        return {"llm_input_messages": trimmed}

    return window_history


def build_context_hook(settings: Settings | None = None):
    """Agent 实际挂的钩子调度器：窗口模式优先，旧的每轮精简作对照。"""

    settings = settings or get_settings()
    if settings.context_window_enabled:
        return build_window_hook(settings)
    return build_slim_hook(settings)
