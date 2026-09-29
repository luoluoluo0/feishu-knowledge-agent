import csv
import logging
from pathlib import Path

from app.config import Settings, get_settings
from app.citations import current_citation_registry
from app.hybrid_retriever import HybridRetriever
from app.milvus_store import MilvusStore
from app.qdrant_store import QdrantStore, RetrievedChunk
from app import query_cache
from app.query_context import get_current_intent, min_score_for_intent


logger = logging.getLogger(__name__)


# Tools 层负责把底层检索封装成“业务工具”。
#
# QdrantStore 关心的是 content_type、item_id、metadata filter。
# PaperSearchTools 关心的是业务动作：
# - 查全部
# - 查 PDF
# - 查 PPT
# - 查文献卡片
# - 查某篇文献
#
# 后面的 Planner / Agent 只调用 tools，不直接碰 Qdrant 细节。


def chunk_display_fields(chunk: RetrievedChunk) -> tuple[str, str, str]:
    """取一块资料的（类型、来源文件、位置）展示字段。

    新旧两套语料字段名不一样，都读一遍，缺的留空：
      旧语料（Qdrant，管 PPT 和文献卡片）—— content_type / source_file / slide
      新语料（Milvus，管论文正文）      —— block_type / title / section / page
    """

    metadata = chunk.metadata
    content_type = metadata.get("content_type") or metadata.get("block_type") or ""
    # block_type 是英文枚举，给 LLM 和来源面板看的展示名用中文
    content_type = {"figure": "论文图表", "table": "表格", "equation": "公式"}.get(
        content_type, content_type
    )
    source_file = metadata.get("source_file") or metadata.get("title") or ""

    locations = []
    if metadata.get("page"):
        locations.append(f"PDF第{metadata.get('page')}页")
    if metadata.get("slide"):
        locations.append(f"PPT第{metadata.get('slide')}页")
    section = metadata.get("section") or ""
    if section and section != "（前置内容）":
        locations.append(section)

    location_text = "，".join(locations) if locations else "位置未标明"
    return content_type, source_file, location_text


def chunk_citation_key(chunk: RetrievedChunk) -> tuple:
    """一块资料的引用身份键。

    必须与 citations.seed_registry_from_history 从历史工具文本里
    解析出的出处键保持同一结构（标题/来源文件/位置——seed 侧拿不到
    metadata，只能从格式化文本里取，所以键只能用这三样），
    这样「一个出处一个号」才能跨回合成立。
    """

    _, source_file, location_text = chunk_display_fields(chunk)
    return (
        str(chunk.metadata.get("title", "") or ""),
        str(source_file or ""),
        str(location_text or ""),
    )


def format_chunk(chunk: RetrievedChunk, index: int) -> str:
    """把一个检索结果格式化成大模型容易阅读的文本。

    index 是本轮请求内的全局引用编号（见 app/citations.py）——多批
    工具结果共用一套编号，同一出处永远同一个号，模型据此写 [n]。
    """

    metadata = chunk.metadata
    content_type, source_file, location_text = chunk_display_fields(chunk)

    source_url = str(metadata.get("source_url") or "")
    if not source_url and metadata.get("item_id"):
        # 飞书完整 token/链接刻意不写进 Milvus；展示时按稳定 item_id
        # 回查 SQLite。查询失败只少一行链接，不影响检索主链。
        from app.feishu_sync.source_registry import lookup_source

        registered = lookup_source(str(metadata.get("item_id"))) or {}
        source_url = str(registered.get("source_url") or "")
    source_url_line = f"来源链接：{source_url}\n" if source_url else ""

    retrieval_lines = []

    if metadata.get("retrieval_method"):
        retrieval_lines.append(f"检索方式：{metadata.get('retrieval_method')}")
    if metadata.get("rrf_score") is not None:
        retrieval_lines.append(f"RRF分数：{float(metadata.get('rrf_score')):.4f}")
    if metadata.get("bm25_rank") is not None:
        retrieval_lines.append(f"BM25排名：{metadata.get('bm25_rank')}")
    if metadata.get("qdrant_rank") is not None:
        retrieval_lines.append(f"Qdrant排名：{metadata.get('qdrant_rank')}")
    if metadata.get("bm25_score") is not None:
        retrieval_lines.append(f"BM25分数：{float(metadata.get('bm25_score')):.4f}")
    if metadata.get("qdrant_score") is not None:
        retrieval_lines.append(f"Qdrant分数：{float(metadata.get('qdrant_score')):.4f}")

    retrieval_text = "".join(f"{line}\n" for line in retrieval_lines)

    # 图表块带图片路径：一行两个读者——LLM 知道证据背后有图可提，
    # citations 解析来源时把路径抄进 source.image 给前端渲染缩略图。
    image_path = str(metadata.get("image_path") or "")
    image_line = f"图片：{image_path}\n" if image_path else ""

    return (
        f"资料{index}\n"
        f"标题：{metadata.get('title', '')}\n"
        f"类型：{content_type}\n"
        f"文献编号：{metadata.get('item_id', '')}\n"
        f"来源文件：{source_file}\n"
        f"{source_url_line}"
        f"位置：{location_text}\n"
        f"{image_line}"
        f"相似度：{chunk.score:.4f}\n"
        f"{retrieval_text}"
        f"内容：{chunk.text}"
    )


def format_tool_result(
    chunks: list[RetrievedChunk],
    *,
    confident: bool | None = None,
    reason: str = "",
) -> str:
    """把检索结果拼成工具返回文本。

    编号用本轮请求的全局引用注册表（app/citations.py）：跨工具调用
    唯一、同一出处同一个号，模型答案里的 [n] 才有唯一指向。不在请求
    链路里（脚本直调、部分单测）没有注册表时，退回批内 1..n，行为
    与从前一致。

    confident=False 时在开头明确告诉模型「这批资料不够」。置信度真正
    落地就在这一句——模型手里有东西就会想办法用上，不明说「没找到」，
    它就会拿弱相关的内容硬凑，那正是幻觉的来源。

    confident 为 None 表示没做判断（未启用重排，或重排调用失败），
    此时保持原来的行为，不加任何提示。
    """

    if not chunks:
        return "没有检索到相关资料。"

    registry = current_citation_registry()
    if registry is not None:
        numbered = [
            (chunk, registry.number_for(chunk_citation_key(chunk)))
            for chunk in chunks
        ]
    else:
        # 与上面保持同一形状：(chunk, 编号)。
        numbered = [(chunk, index) for index, chunk in enumerate(chunks, start=1)]

    body = "\n\n".join(
        format_chunk(chunk, index)
        for chunk, index in numbered
    )

    if confident is False:
        return (
            "⚠️ 没有检索到足够相关的资料。\n"
            f"（{reason}）\n"
            "下面列出的是相似度最高的一些片段，但很可能与问题无关。\n"
            "如果它们确实回答不了这个问题，请直接告诉用户「语料库里没有相关内容」，"
            "不要勉强拼凑，也不要补充语料之外的背景知识。\n\n"
            f"{body}"
        )

    return body


def merge_by_rank(
    primary: list[RetrievedChunk],
    secondary: list[RetrievedChunk],
    limit: int,
) -> list[RetrievedChunk]:
    """把两批结果按各自的排名交替合并。

    不做统一重排：两边的分数根本不是一回事——一边是重排模型的
    相关性分（0~1，实测语料内的都在 0.95 以上），另一边是双塔余弦
    （挤在 0.5~0.6）。混在一起排序等于让尺度大的那一边通吃。

    交替插入至少保证两边都进得来，不会因为分数尺度差异被整体挤掉。
    """

    merged: list[RetrievedChunk] = []
    for index in range(max(len(primary), len(secondary))):
        if index < len(primary):
            merged.append(primary[index])
        if index < len(secondary):
            merged.append(secondary[index])
    return merged[:limit]


PROJECT_DIR = Path(__file__).resolve().parent.parent
LITERATURE_INDEX_FILE = PROJECT_DIR / "data" / "metadata" / "literature_index_auto.csv"


def format_literature_rows(rows: list[dict]) -> str:
    """把文献索引表里的精确匹配结果格式化成工具返回文本。"""

    if not rows:
        return "没有在文献索引表中找到匹配的文献。"

    lines = [f"共找到 {len(rows)} 篇文献："]
    for index, row in enumerate(rows, start=1):
        lines.append(
            "\n".join(
                [
                    f"{index}. 文献编号：{row.get('item_id', '')}",
                    f"   标题：{row.get('title', '')}",
                    f"   阅读整理者：{row.get('reader', '')}",
                    f"   DOI：{row.get('doi', '') or '未填写'}",
                    f"   PDF文件：{row.get('paper_file', '') or '未匹配'}",
                    f"   PPT文件：{row.get('ppt_file', '') or '未匹配'}",
                    f"   主题：{row.get('theme', '') or '未填写'}",
                    f"   状态：{row.get('status', '') or '未填写'}",
                ]
            )
        )

    return "\n".join(lines)


def read_literature_index() -> list[dict]:
    """读取文献索引表。结构化查询工具都从这里查，不走向量检索。"""

    if not LITERATURE_INDEX_FILE.exists():
        # 开源版可从空库启动；飞书自动同步不依赖旧的人工 CSV 索引。
        logger.info("旧文献索引不存在，按空索引运行：%s", LITERATURE_INDEX_FILE)
        return []

    with LITERATURE_INDEX_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def normalize_item_id(item_id: str) -> str:
    """把 31 / 031 这类编号统一成 031。"""

    item_id = item_id.strip()
    if item_id.isdigit():
        return item_id.zfill(3)
    return item_id


def normalize_status(status: str) -> str:
    """把用户常说的中文状态转成索引表里的英文状态。"""

    status = status.strip().lower()
    aliases = {
        "资料齐全": "ready",
        "齐全": "ready",
        "ready": "ready",
        "缺pdf": "missing_pdf",
        "缺 pdf": "missing_pdf",
        "missing_pdf": "missing_pdf",
        "缺ppt": "missing_ppt",
        "缺 ppt": "missing_ppt",
        "missing_ppt": "missing_ppt",
        "只有笔记": "note_only",
        "note_only": "note_only",
    }
    return aliases.get(status, status)


def format_semantic_literature_matches(matches: list[dict]) -> str:
    """格式化语义召回后的文献列表。"""

    if not matches:
        return "没有从向量库中检索到相关文献。"

    lines = [f"向量库语义检索后，共召回 {len(matches)} 篇候选文献："]
    for index, match in enumerate(matches, start=1):
        row = match["row"]
        content_types = "、".join(sorted(match["content_types"])) or "未标明"
        source_files = "、".join(sorted(match["source_files"])) or "未标明"
        lines.append(
            "\n".join(
                [
                    f"{index}. 文献编号：{row.get('item_id', '')}",
                    f"   标题：{row.get('title', '')}",
                    f"   阅读整理者：{row.get('reader', '')}",
                    f"   DOI：{row.get('doi', '') or '未填写'}",
                    f"   PDF文件：{row.get('paper_file', '') or '未匹配'}",
                    f"   PPT文件：{row.get('ppt_file', '') or '未匹配'}",
                    f"   主题：{row.get('theme', '') or '未填写'}",
                    f"   状态：{row.get('status', '') or '未填写'}",
                    f"   最高相似度：{match['score']:.4f}",
                    f"   命中资料类型：{content_types}",
                    f"   命中来源文件：{source_files}",
                ]
            )
        )

    return "\n".join(lines)


class PaperSearchTools:
    """论文分享项目的检索工具集合。"""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

        # 论文正文走 Milvus：MinerU 解析的新语料 + 内建 BM25 + 跨语言
        # 翻译 + 重排与置信度判断。
        #
        # PPT 和文献卡片仍留在 Qdrant —— 新语料是从 PDF 解析出来的，
        # 里面没有这两类内容（旧语料有 2556 条 slide、104 条
        # literature_card）。整个换掉会让 PPT 检索和卡片检索失效，
        # 而 Agent 的提示词明确要求按内容类型分流。
        self.paper_store = MilvusStore(self.settings)
        self.store = QdrantStore(self.settings)
        self.hybrid_retriever = HybridRetriever(self.settings)

    def _paper_retrieve(
        self,
        query: str,
        top_k: int,
        *,
        item_id: str | None = None,
    ) -> tuple[list[RetrievedChunk], bool | None, str]:
        """查论文正文，返回 (结果, 置信度, 说明)。

        只有真的做了重排才回传置信度。没重排时 confident 是默认的
        True，回传的却是 None——表示「没判过」，别让调用方误以为
        这是「判过且置信」。

        rerank 门槛按意图分档：意图由本次请求的预处理写入 ContextVar
        （见 app/query_context.py），总结类问题用低门槛，事实类维持
        0.8。检索层出错时返回空结果加一句说明，不抛异常：换库期间
        不该因为 Milvus 连不上就让整个 Agent 崩掉。
        """

        intent = get_current_intent()
        min_score = min_score_for_intent(intent, self.settings)
        if intent:
            logger.info("rerank 门槛按意图分档：%s -> %.2f", intent, min_score)

        # 参考文献块按意图过滤。内容型意图下排除：重排对参考文献区打分
        # 虚高（满屏"方法""研究"字样），实测它能把正文方法章节挤到第 3、
        # 第 5 位，模型却拿它当主要证据。元数据类问题（"012 引用了哪些
        # 文献"）参考文献块本身就是答案，保留；无意图（测试/脚本直调，
        # 空串）保持旧行为不过滤，与门槛分档的回落策略一致。
        include_reference = not intent or intent == "metadata_query"

        # 查询缓存：相同问题（含 item_id/门槛/条数/是否含参考文献）直接
        # 复用上次检索产出，省掉翻译→嵌入→检索→重排约 4 秒和全部 API 费用。
        cache_key = None
        if self.settings.query_cache_enabled:
            cache_key = query_cache.make_key(
                query, item_id, min_score, top_k, include_reference=include_reference
            )
            cached = query_cache.get(cache_key)
            if cached is not None:
                logger.info("查询缓存命中：%s", query[:40])
                return cached

        try:
            result = self.paper_store.retrieve(
                query,
                top_k=top_k,
                item_id=item_id,
                expand_parents=self.settings.parent_expand_enabled,
                min_score=min_score,
                include_reference=include_reference,
            )
        except Exception as exc:
            return [], None, f"论文检索失败：{type(exc).__name__}：{str(exc)[:80]}"

        confident = result.confident if result.reranked else None
        if item_id and result.chunks and confident is False:
            # 置信度门槛是按「语料里到底有没有答案」校准的（事实型问题）。
            # item_id 已经把范围锁到用户点名的那篇文献，片段也确实来自
            # 这一篇——此时再挂「没有检索到足够相关的资料」，模型就会
            # 对着正确的文献说「未找到」。总结、提纲、对比这类宽泛问题
            # 的重排分天然偏低，不构成拒答依据。
            confident = True

        retrieved = (result.chunks, confident, result.reason)
        if cache_key is not None:
            query_cache.put(cache_key, retrieved)
        return retrieved

    def _paper_result(self, query: str, top_k: int, *, item_id: str | None = None) -> str:
        """查论文正文，格式化后返回。

        纯向量与混合检索在 Milvus 这边合并成了一条路径——两者的差别
        已经由检索层内部处理（BM25 + RRF + 重排），Tools 层不必再分。
        保留两个方法名只是为了不破坏 Agent 已有的工具集。
        """

        chunks, confident, reason = self._paper_retrieve(query, top_k, item_id=item_id)
        return format_tool_result(chunks, confident=confident, reason=reason)

    def list_literature_by_reader(self, reader: str) -> str:
        """按阅读整理者精确列出文献，适合“某某讲了哪些论文”这类问题。"""

        reader = reader.strip()
        if not reader:
            return "请提供阅读整理者姓名。"

        rows = [
            row
            for row in read_literature_index()
            if reader in (row.get("reader") or "")
        ]

        rows.sort(key=lambda row: row.get("item_id", ""))
        return format_literature_rows(rows)

    def semantic_search_literature(self, query: str, top_k: int = 40) -> str:
        """用向量库语义检索相关文献，适合任意主题、方向、概念的文献列表问题。"""

        query = query.strip()
        if not query:
            return "请提供要查询的主题、方向或关键词。"

        rows_by_item_id = {
            row.get("item_id", ""): row
            for row in read_literature_index()
            if row.get("item_id")
        }
        # 语义检索用纯向量，不走向 BM25 那一路——这个方法要的是
        # 「按主题方向召回一批文献」，关键词匹配帮不上忙。
        try:
            chunks = self.paper_store.search(query, top_k=top_k)
        except Exception as exc:
            return f"文献语义检索失败：{type(exc).__name__}：{str(exc)[:100]}"

        matches_by_item_id: dict[str, dict] = {}
        for chunk in chunks:
            metadata = chunk.metadata
            item_id = metadata.get("item_id", "")
            if not item_id:
                continue

            row = rows_by_item_id.get(item_id, dict(metadata))
            match = matches_by_item_id.setdefault(
                item_id,
                {
                    "row": row,
                    "score": chunk.score,
                    "content_types": set(),
                    "source_files": set(),
                },
            )
            match["score"] = max(match["score"], chunk.score)
            if metadata.get("content_type"):
                match["content_types"].add(metadata["content_type"])
            if metadata.get("source_file"):
                match["source_files"].add(metadata["source_file"])

        matches = sorted(
            matches_by_item_id.values(),
            key=lambda match: match["score"],
            reverse=True,
        )

        return format_semantic_literature_matches(matches)

    def list_literature_by_status(self, status: str) -> str:
        """按资料状态列出文献，比如 ready、missing_pdf、missing_ppt。"""

        status = normalize_status(status)
        if not status:
            return "请提供状态：ready、missing_pdf、missing_ppt 或 note_only。"

        rows = [
            row
            for row in read_literature_index()
            if (row.get("status") or "") == status
        ]
        rows.sort(key=lambda row: row.get("item_id", ""))
        return format_literature_rows(rows)

    def get_literature_by_item_id(self, item_id: str) -> str:
        """按文献编号精确获取一篇文献的元信息。"""

        item_id = normalize_item_id(item_id)
        rows = [
            row
            for row in read_literature_index()
            if row.get("item_id") == item_id
        ]
        return format_literature_rows(rows)

    def list_missing_files(self) -> str:
        """列出所有资料不齐全的文献。"""

        rows = [
            row
            for row in read_literature_index()
            if (row.get("status") or "") != "ready"
        ]
        rows.sort(key=lambda row: (row.get("status", ""), row.get("item_id", "")))
        return format_literature_rows(rows)

    def _legacy_retrieve(
        self,
        query: str,
        top_k: int,
        *,
        item_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """查旧语料（PPT、文献卡片，走 Qdrant），失败降级为空结果。

        和 _paper_retrieve 的策略一致：检索层出错返回空列表加日志，
        不抛异常——Qdrant 挂了不该连带整个 Agent 或 PlannerAgent 崩掉，
        论文正文那条路（Milvus）还活着就还能答。
        """

        try:
            if item_id:
                return self.store.search_by_item(query, item_id=item_id, top_k=top_k)
            return self.store.search(query, top_k=top_k)
        except Exception as exc:
            logger.warning(
                "旧语料检索失败（Qdrant），降级为空结果：%s", exc
            )
            return []

    def search_all(self, query: str, top_k: int = 5) -> str:
        """全库检索：不知道该查 PDF、PPT 还是文献卡片时使用。"""
        paper_chunks, confident, reason = self._paper_retrieve(query, top_k)
        others = self._legacy_retrieve(query, top_k)
        # 置信度只反映论文正文那一部分——PPT 和卡片走的还是 Qdrant，
        # 没有重排，也就没有可比的判断依据。所以论文没找到时提示一句，
        # 但下面的内容里可能还有 PPT 和卡片，让模型自己看着办。
        return format_tool_result(
            merge_by_rank(paper_chunks, others, top_k),
            confident=confident,
            reason=reason,
        )

    def search_paper(
        self,
        query: str,
        top_k: int = 5,
        *,
        item_id: str | None = None,
    ) -> str:
        """只查 PDF 正文：适合研究问题、方法、理论、结论。

        item_id 用于计划里已确定文献编号的场景（组会提纲、对比文献），
        限定范围后总结类问题的重排分不参与拒答判断（见 _paper_retrieve）。
        """

        return self._paper_result(query, top_k, item_id=item_id)

    def search_by_item(self, query: str, item_id: str, top_k: int = 5) -> str:
        """只查某一篇文献：适合用户明确指定第几篇文献时使用。"""
        item_id = normalize_item_id(item_id)
        paper_chunks, confident, reason = self._paper_retrieve(
            query, top_k, item_id=item_id
        )
        others = self._legacy_retrieve(query, top_k, item_id=item_id)
        return format_tool_result(
            merge_by_rank(paper_chunks, others, top_k),
            confident=confident,
            reason=reason,
        )

    def hybrid_search_ppt(self, query: str, top_k: int = 5) -> str:
        """混合检索 PPT：适合查组会汇报结构、幻灯片内容、PPT 重点。"""
        chunks = self.hybrid_retriever.search_ppt(query, top_k=top_k)
        return format_tool_result(chunks)

    def hybrid_search_literature_card(self, query: str, top_k: int = 5) -> str:
        """混合检索文献卡片：适合查标题、整理者、DOI、附件、文献基础信息。"""
        chunks = self.hybrid_retriever.search_literature_card(query, top_k=top_k)
        return format_tool_result(chunks)
