"""单轮请求内的引用编号注册表。

背景：一次问答里 Agent 可能调多次检索工具，原来每批工具结果的块编号
都从「资料1」重新数。模型想在答案里说「这个结论出自第 3 块资料」时，
上下文里同时存在三个「资料3」——引用无从对起，前端来源面板也没法
和答案里的编号关联。

解法：给每次请求发一个全局编号注册表。同一出处（同一篇、同一位置）
在一个请求内只占一个号，先到先得；后面任何工具再检索到同一出处，
复用旧号。工具格式化结果时取号，答案里的 [n]、done.sources 里的 n
三边天然对齐。

生命周期与意图 ContextVar 同款：prepare_question（与后续工具调用同
线程同调用栈）里重置，请求结束随线程上下文丢弃，不跨请求累积。
"""

import contextvars
import re


class CitationRegistry:
    """出处 → 编号 的映射。编号按首次出现顺序连续分配。

    多轮会话里历史工具消息的「资料N」一直在模型上下文里，新回合
    若从 1 重新编起，同一个 [3] 就指代不明。start_index 与 initial_map
    配合 seed_registry_from_history 使用：出处映射也一并种回，同一
    出处无论哪一轮再检索到，拿到的都是同一个号——整个会话内
    「一个出处一个号」，答案里的 [n] 永远唯一指向。
    """

    def __init__(self, start_index: int = 1, initial_map: dict | None = None) -> None:
        self._numbers: dict[tuple, int] = dict(initial_map or {})
        # 新号从「起点」和「已占用最大号 +1」里取大者，绝不复用旧号
        self._next = max(start_index, max(self._numbers.values(), default=0) + 1)

    def number_for(self, key: tuple) -> int:
        """取这个出处的编号，没有就分配下一个。"""

        if key not in self._numbers:
            self._numbers[key] = self._next
            self._next += 1
        return self._numbers[key]


_registry_var: contextvars.ContextVar[CitationRegistry | None] = (
    contextvars.ContextVar("citation_registry", default=None)
)


def reset_citation_registry(start_index: int = 1, initial_map: dict | None = None):
    """开一个新注册表，编号从 start_index 起。

    每个请求开始时调一次。传 start_index 与 initial_map 即实现跨回合
    续号（见 seed_registry_from_history）。
    """

    registry = CitationRegistry(start_index, initial_map)
    _registry_var.set(registry)
    return registry


def current_citation_registry() -> CitationRegistry | None:
    """当前请求的注册表；不在请求链路里（脚本、测试直调）时是 None。"""

    return _registry_var.get()


def clear_citation_registry() -> None:
    """清掉注册表。测试隔离用——防止用例之间通过线程上下文互相漏号。"""

    _registry_var.set(None)


# ----------------------------------------------------------------------
# 工具结果文本 → 来源信息的解析。放在这个模块里，api.py（流式面板）
# 和下面的 seed_registry_from_history（历史续号）共用同一套解析，
# 保证「种回去的出处键」和「工具格式化时的出处键」永远一致。
# ----------------------------------------------------------------------

def get_text_field(block: str, label: str) -> str:
    """取「标签：值」行里的值，没有就返回空串。

    锚定行首（容忍缩进）：内容正文里也可能出现「标题：」这类词，
    不锚定会取到正文里的假字段。语义文献列表的块首行自带序号前缀
    （「1. 文献编号：032」），所以还要容忍「N. 」前缀——否则每个
    块的第一个字段永远取不到（2026-09-21 修：语义检索来源的
    item_id 全空、引用面板对不上编号，根因就在这）。
    """

    match = re.search(rf"^\s*(?:\d+\.\s*)?{re.escape(label)}：(.*)$", block, re.MULTILINE)
    return match.group(1).strip() if match else ""


def split_source_blocks(text: str) -> list[str]:
    """把工具返回的大段文本切成一条条来源。

    tools.py 里的返回文本主要有两种形态：
    - 资料1 / 资料2：向量检索命中的 chunk
    - 1. 文献编号：结构化文献索引里的文献条目

    注意只允许「带序号前缀」的文献编号行作为切分点：检索 chunk 块的
    字段行里也有裸的「文献编号：045」（2026-09-21 起 format_chunk
    会输出它），不要求序号前缀会把每个检索块从中间劈成两半。
    """

    blocks = re.split(r"\n(?=\d+\.\s*文献编号：|资料\d+\b)", text)
    return [block.strip() for block in blocks if block.strip()]


def extract_source_blocks_with_number(tool_name: str, text: str) -> list[dict]:
    """从工具返回文本提取来源清单，块头「资料N」的 N 记入 n 字段。

    n 供来源面板与答案里的 [n] 对齐；结构化索引条目没有编号，是 None。
    api.extract_sources_from_tool_text 直接委托本函数——面板解析与
    seed 历史续号用的是同一套，出处键永远一致。
    """

    if not text or "没有检索到" in text or "没有在文献索引表中找到" in text:
        return []

    sources = []
    for block in split_source_blocks(text):
        title = get_text_field(block, "标题")
        item_id = get_text_field(block, "文献编号")
        content_type = get_text_field(block, "类型") or get_text_field(block, "命中资料类型")
        source_file = (
            get_text_field(block, "来源文件")
            or get_text_field(block, "命中来源文件")
            or get_text_field(block, "PDF文件")
            or get_text_field(block, "PPT文件")
        )
        if not title and not item_id and not source_file:
            continue

        match = re.match(r"资料(\d+)", block)
        sources.append(
            {
                "n": int(match.group(1)) if match else None,
                "tool": tool_name,
                "item_id": item_id,
                "title": title,
                "content_type": content_type,
                "source_file": source_file,
                "source_url": get_text_field(block, "来源链接") or None,
                "location": get_text_field(block, "位置"),
                # 图表块的图片路径（format_chunk 输出的「图片：」行），
                # 前端据此渲染缩略图；非图表块为 None。
                "image": get_text_field(block, "图片") or None,
                "score": get_text_field(block, "相似度")
                or get_text_field(block, "最高相似度"),
                "retrieval_method": get_text_field(block, "检索方式"),
                "rrf_score": get_text_field(block, "RRF分数"),
                "bm25_rank": get_text_field(block, "BM25排名"),
                "qdrant_rank": get_text_field(block, "Qdrant排名"),
                "bm25_score": get_text_field(block, "BM25分数"),
                "qdrant_score": get_text_field(block, "Qdrant分数"),
            }
        )
    return sources


def extract_all_tool_texts(messages) -> list[tuple[str, str]]:
    """取消息历史里全部 ToolMessage（工具名, 原文）。

    编号按会话续接后（见 seed_registry_from_history）跨回合不撞号，
    所以来源面板的恢复不必再局限于最后一个回合——模型引哪个回合
    的号都能对上。
    """

    return [
        (str(getattr(message, "name", "") or "tool"), str(getattr(message, "content", "") or ""))
        for message in messages
        if getattr(message, "type", "") == "tool"
    ]


def seed_registry_from_history(agent, config) -> list[tuple[str, str]]:
    """按会话历史给注册表续号并种回出处映射，返回历史工具消息文本。

    从 checkpointer 读全部历史工具消息：
    - 出处映射：解析每个「资料N」块的（标题/来源文件/位置），把
      出处 → N 种回注册表。同一出处以后再被检索到，拿到的还是 N。
    - 计数起点：全部历史里最大 N + 1，新出处接着发号。

    这样整个会话内「一个出处一个号」：模型复用历史检索作答时引的
    老号、新检索发的新号，都能在来源面板里对上。没有历史（新会话）
    或读状态失败时从 1 起，行为同旧版。返回历史文本供面板合并复用，
    避免二次读状态。
    """

    texts: list[tuple[str, str]] = []
    initial_map: dict = {}
    max_number = 0
    try:
        state = agent.get_state(config)
        texts = extract_all_tool_texts((state.values or {}).get("messages", []))
        for tool_name, text in texts:
            for source in extract_source_blocks_with_number(tool_name, text):
                number = source.get("n")
                if number is None:
                    continue
                max_number = max(max_number, number)
                # 出处键与 tools.chunk_citation_key 同构（标题/来源文件/位置）
                key = (
                    source.get("title") or "",
                    source.get("source_file") or "",
                    source.get("location") or "",
                )
                # 同一出处跨回合重复出现时，保留最早的号
                initial_map.setdefault(key, number)
    except Exception:
        texts = []
        initial_map = {}
        max_number = 0

    reset_citation_registry(max_number + 1, initial_map)
    return texts
