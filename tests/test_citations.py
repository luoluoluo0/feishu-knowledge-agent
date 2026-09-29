"""引用编号链路的定向测试：注册表、格式化取号、来源提取带编号、排序。

只测纯逻辑，不起服务、不连 Milvus。api 模块的导入沿用
test_api_rewrite_fields.py 的先例。
"""

import pytest

from app import api as api_module
from app.agent import AGENT_SYSTEM_PROMPT
from app.citations import (
    CitationRegistry,
    clear_citation_registry,
    reset_citation_registry,
)
from app.planner_agent import PlannerAgent
from app.qdrant_store import RetrievedChunk
from app.tools import chunk_citation_key, format_tool_result


@pytest.fixture(autouse=True)
def _clean_registry():
    """每个用例结束后清掉注册表，防止编号漏进同线程的后续用例。"""

    yield
    clear_citation_registry()


def make_chunk(text: str, **metadata) -> RetrievedChunk:
    base = {"title": "某论文", "source_file": "paper.pdf", "page": 3}
    base.update(metadata)
    return RetrievedChunk(score=0.9, text=text, metadata=base)


class TestCitationRegistry:
    def test_numbers_dense_and_stable(self):
        registry = CitationRegistry()
        assert registry.number_for(("a",)) == 1
        assert registry.number_for(("b",)) == 2
        # 同一处所再取号，必须还是原来的号
        assert registry.number_for(("a",)) == 1
        assert registry.number_for(("c",)) == 3

    def test_fresh_registry_restarts_at_one(self):
        first = CitationRegistry()
        first.number_for(("x",))
        second = CitationRegistry()
        assert second.number_for(("x",)) == 1


class TestFormatWithRegistry:
    def test_fallback_without_registry_keeps_local_numbering(self):
        reset_citation_registry()
        clear_citation_registry()
        chunks = [make_chunk("第一段"), make_chunk("第二段")]
        text = format_tool_result(chunks)
        assert "资料1" in text and "资料2" in text
        assert "资料3" not in text

    def test_cross_batch_reuses_same_number(self):
        reset_citation_registry()
        shared = make_chunk("共享块", page=1)
        first = format_tool_result([shared, make_chunk("独有块", page=2)])
        second = format_tool_result([make_chunk("另一块", page=3), shared])

        # 共享块在两批结果里编号一致；同批内编号不重复
        assert "资料1" in first and "资料2" in first
        assert "资料1" in second and "资料3" in second
        # 第二批没有自己的「资料2」（那是第一批独有块的号）
        assert "资料2" not in second

    def test_numbers_are_dense_from_one(self):
        registry = reset_citation_registry()
        format_tool_result(
            [make_chunk("a", page=1), make_chunk("b", page=2)]
        )
        format_tool_result([make_chunk("c", page=3)])
        # 三块资料三个号，按首次出现顺序连续分配——
        # 这是「来源面板按编号排序后恰为 [1][2][3] 前缀」的前提。
        assert sorted(registry._numbers.values()) == [1, 2, 3]


class TestCitationKey:
    def test_same_location_same_key_regardless_of_text(self):
        one = make_chunk("前半段内容", page=3)
        two = make_chunk("后半段内容", page=3)
        assert chunk_citation_key(one) == chunk_citation_key(two)

    def test_different_page_different_key(self):
        one = make_chunk("同样内容", page=3)
        two = make_chunk("同样内容", page=4)
        assert chunk_citation_key(one) != chunk_citation_key(two)


class TestSourceExtraction:
    def test_extract_carries_citation_number(self):
        text = (
            "资料1\n标题：论文A\n类型：paper_text\n来源文件：a.pdf\n"
            "位置：PDF第2页\n相似度：0.9100\n内容：一些内容\n\n"
            "资料2\n标题：论文B\n类型：paper_text\n来源文件：b.pdf\n"
            "位置：PDF第5页\n相似度：0.8800\n内容：另一些内容"
        )
        sources = api_module.extract_sources_from_tool_text("search_paper", text)
        assert [source["n"] for source in sources] == [1, 2]

    def test_index_rows_have_no_number(self):
        text = (
            "共找到 1 篇文献：\n"
            "1. 文献编号：031\n   标题：某文献\n   阅读整理者：张三\n   DOI：\n"
            "   PDF文件：x.pdf\n   PPT文件：\n   主题：\n   状态：ready"
        )
        sources = api_module.extract_sources_from_tool_text(
            "get_literature_by_item_id", text
        )
        assert len(sources) == 1
        assert sources[0]["n"] is None

    def test_feishu_source_url_is_extracted(self, monkeypatch):
        from app.milvus_store import RetrievedChunk
        from app.tools import format_chunk

        monkeypatch.setattr(
            "app.feishu_sync.source_registry.lookup_source",
            lambda item_id: {
                "source_url": "https://example.feishu.cn/docx/example"
            },
        )
        chunk = RetrievedChunk(
            score=0.91,
            text="自动同步正文",
            metadata={"item_id": "1234567890abcdef", "title": "飞书周报"},
        )

        text = format_chunk(chunk, 1)
        sources = api_module.extract_sources_from_tool_text("search_paper", text)

        assert "来源链接：https://example.feishu.cn/docx/example" in text
        assert sources[0]["source_url"] == "https://example.feishu.cn/docx/example"

    def test_order_puts_numbered_first_sorted(self):
        sources = [
            {"n": None, "title": "索引条目"},
            {"n": 2, "title": "B"},
            {"n": 1, "title": "A"},
            {"n": None, "title": "另一个条目"},
        ]
        ordered = api_module.order_sources_by_citation(sources)
        assert [source.get("n") for source in ordered] == [1, 2, None, None]
        assert [source["title"] for source in ordered] == ["A", "B", "索引条目", "另一个条目"]

    def test_dedupe_keeps_first_occurrence(self):
        sources = [
            {"n": 1, "title": "A", "item_id": "", "source_file": "a.pdf", "location": "L1"},
            {"n": 1, "title": "A", "item_id": "", "source_file": "a.pdf", "location": "L1"},
        ]
        assert len(api_module.dedupe_sources(sources)) == 1


def test_prompts_document_citation_format():
    # 提示词必须教会模型写 [n] 并说明编号来源，否则整条链路没有源头。
    # [7]/「资料7」是提示词里约定的正例写法。
    assert "[7]" in AGENT_SYSTEM_PROMPT and "资料7" in AGENT_SYSTEM_PROMPT

    # 开源版的主角色必须是通用飞书知识库 Agent；论文阅读只是兼容场景，
    # 不能让同步普通制度、产品或项目文档的用户被论文角色带偏。
    from app.intent import INTENT_SYSTEM_PROMPT
    from app.planner import PLANNER_SYSTEM_PROMPT

    assert "飞书知识库" in AGENT_SYSTEM_PROMPT
    assert "飞书知识库" in INTENT_SYSTEM_PROMPT
    assert "飞书知识库" in PLANNER_SYSTEM_PROMPT

    from types import SimpleNamespace

    from app.planner import PlanResult

    # build_final_prompt 现在会读 settings 里的拼装瘦身配置，
    # 传一个最小命名空间即可，不拖真实配置。
    prompt = PlannerAgent.build_final_prompt(
        SimpleNamespace(settings=SimpleNamespace(planner_prompt_top_k_per_step=0)),
        "测试问题",
        PlanResult(task_type="普通问答"),
        [],
    )
    assert "[7]" in prompt and "资料7" in prompt


def test_extract_all_tool_texts_and_seed_reuse_numbers():
    from types import SimpleNamespace

    from app.citations import (
        clear_citation_registry,
        current_citation_registry,
        extract_all_tool_texts,
        seed_registry_from_history,
    )
    from app.tools import chunk_citation_key

    def tool(name, text):
        return SimpleNamespace(type="tool", name=name, content=text)

    def plain(kind):
        return SimpleNamespace(type=kind, name="", content="", tool_calls=None)

    block = (
        "资料6\n标题：论文甲\n类型：paper_text\n来源文件：a.pdf\n"
        "位置：PDF第3页\n相似度：0.9000\n内容：正文"
    )
    messages = [
        plain("human"),
        tool("search_paper", block),
        plain("ai"),
        tool("search_by_item", "资料7\n标题：论文甲\n类型：paper_text\n来源文件：a.pdf\n位置：PDF第3页\n相似度：0.8800\n内容：正文"),
    ]
    texts = extract_all_tool_texts(messages)
    assert len(texts) == 2

    fake_agent = SimpleNamespace(
        get_state=lambda config: SimpleNamespace(values={"messages": messages})
    )
    seed_registry_from_history(fake_agent, {})

    # 同一出处跨回合重复检索：映射里保留了最早的号 6，
    # 新回合再遇到它必须还是 6，而不是发新号
    chunk = RetrievedChunk(
        score=0.9,
        text="正文",
        metadata={"title": "论文甲", "source_file": "a.pdf", "page": 3},
    )
    assert current_citation_registry().number_for(chunk_citation_key(chunk)) == 6

    # 全新出处接在最大号 7 之后
    new_chunk = RetrievedChunk(
        score=0.9,
        text="别的",
        metadata={"title": "论文乙", "source_file": "b.pdf", "page": 1},
    )
    assert current_citation_registry().number_for(chunk_citation_key(new_chunk)) == 8
    clear_citation_registry()

    # 全程没有工具消息 → 空
    assert extract_all_tool_texts([plain("human"), plain("ai")]) == []


def test_seed_registry_continues_numbering_from_history():
    from types import SimpleNamespace

    from app.citations import (
        clear_citation_registry,
        current_citation_registry,
        seed_registry_from_history,
    )

    def tool(name, text):
        return SimpleNamespace(type="tool", name=name, content=text)

    history_messages = [
        tool(
            "search_paper",
            "资料14\n标题：旧\n类型：paper_text\n来源文件：old.pdf\n位置：PDF第1页\n相似度：0.9\n内容：x\n\n"
            "资料15\n标题：旧2\n类型：paper_text\n来源文件：old2.pdf\n位置：PDF第2页\n相似度：0.9\n内容：y",
        )
    ]
    fake_agent = SimpleNamespace(
        get_state=lambda config: SimpleNamespace(
            values={"messages": history_messages}
        )
    )

    texts = seed_registry_from_history(fake_agent, {})
    assert len(texts) == 1
    # seed 设置的注册表：新回合的编号从历史最大 15 之后续接，不撞号
    assert current_citation_registry().number_for(("新出处",)) == 16

    # 历史为空（新会话）→ 从 1 起
    empty_agent = SimpleNamespace(
        get_state=lambda config: SimpleNamespace(values={"messages": []})
    )
    seed_registry_from_history(empty_agent, {})
    assert current_citation_registry().number_for(("新出处",)) == 1

    # 读状态抛异常 → 降级从 1 起，不炸
    def boom(config):
        raise RuntimeError("state 不可用")

    seed_registry_from_history(SimpleNamespace(get_state=boom), {})
    assert current_citation_registry().number_for(("新出处",)) == 1
    clear_citation_registry()


class TestCapSourcesWithCitations:
    def _sources(self, count):
        return [
            {"n": n, "title": f"论文{n}", "item_id": "", "source_file": f"f{n}", "location": "L"}
            for n in range(1, count + 1)
        ]

    def test_within_cap_unchanged(self):
        from app.api import cap_sources_with_citations

        sources = self._sources(15)
        assert len(cap_sources_with_citations(sources, "答案 [3]", cap=20)) == 15

    def test_cited_source_survives_truncation(self):
        from app.api import cap_sources_with_citations

        sources = self._sources(30)
        capped = cap_sources_with_citations(sources, "结论 [28]", cap=20)
        numbers = [source["n"] for source in capped]
        assert len(capped) == 20
        assert 28 in numbers, "被引用的高编号必须保留"
        assert numbers == sorted(numbers)

    def test_no_citations_takes_first_cap(self):
        from app.api import cap_sources_with_citations

        sources = self._sources(30)
        capped = cap_sources_with_citations(sources, "没有引用的答案", cap=20)
        assert [source["n"] for source in capped] == list(range(1, 21))

    def test_cited_count_exceeding_cap_wins(self):
        from app.api import cap_sources_with_citations

        sources = self._sources(30)
        answer = " ".join(f"[{n}]" for n in range(1, 26))
        capped = cap_sources_with_citations(sources, answer, cap=20)
        # 引用优先于上限：25 个被引来源全部保留
        assert len(capped) == 25


def test_figure_chunk_round_trips_image_to_source():
    """图表块的 image_path 要走通「证据文本 → 来源解析 → source.image」。

    方案 A（图表问答）的前置：路径写进证据文本给 LLM 看，来源解析
    再抄进 source.image 给前端渲染缩略图；非图表块不得带 image。
    """

    from app.milvus_store import RetrievedChunk as MilvusChunk
    from app.citations import extract_source_blocks_with_number
    from app.tools import format_chunk

    figure = MilvusChunk(
        score=0.87,
        text="图 1 利用 Nvivo编码得出的关联因素示意",
        metadata={
            "title": "可行信息能力研究",
            "item_id": "002",
            "block_type": "figure",
            "image_path": "data/processed/figures/002_fig01.png",
            "page": 6,
        },
    )
    text = format_chunk(figure, 1)
    assert "图片：data/processed/figures/002_fig01.png" in text
    assert "类型：论文图表" in text  # 英文枚举转中文展示

    sources = extract_source_blocks_with_number("hybrid_search_literature", text)
    assert len(sources) == 1
    assert sources[0]["image"] == "data/processed/figures/002_fig01.png"

    plain = MilvusChunk(
        score=0.9,
        text="正文内容",
        metadata={"title": "论文甲", "item_id": "003", "block_type": "text"},
    )
    plain_sources = extract_source_blocks_with_number("x", format_chunk(plain, 1))
    assert plain_sources[0]["image"] is None
