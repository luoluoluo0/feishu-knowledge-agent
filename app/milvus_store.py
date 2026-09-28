import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from app.config import Settings, get_settings
from app.embeddings import build_embeddings, estimate_char_limit
from app.query_translation import needs_translation, translate_to_english
from app.rerank import rerank_documents


# Milvus 向量库接入层，用于替换 Qdrant。
#
# 存的是「子块」：检索单元，带向量。
# 父块不参与向量检索，只在命中后按 id 查回，所以存在本地 JSON 里
# （见 ParentStore），既不进 Milvus 也不生成向量。
#
# 本模块暂时和 qdrant_store.py 并存，不改动线上检索链路。


logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
PARENT_STORE_FILE = PROJECT_DIR / "data" / "processed" / "parents.json"

VECTOR_DIM = 1024

# VARCHAR 的 max_length 是字符数上限，按实测值留了数倍余量。
MAX_TEXT = 16384
MAX_HTML = 8192
MAX_SECTION = 512
MAX_TITLE = 1024
MAX_ID = 64
MAX_SHORT = 64
MAX_PATH = 256

# 需要写入 Milvus 的标量字段：(字段名, max_length)
SCALAR_FIELDS = [
    ("item_id", 16),
    ("title", MAX_TITLE),
    ("reader", MAX_SHORT),
    ("section", MAX_SECTION),
    ("text", MAX_TEXT),
    ("html", MAX_HTML),
    ("parent_id", MAX_ID),
    ("block_type", 32),
    ("label", MAX_SHORT),
    ("image_path", MAX_PATH),
    ("is_reference", None),  # BOOL
    ("page", None),  # INT32
    ("page_end", None),  # INT32
]

# 检索和查询时要取回的字段。
#
# chunk_id 是主键，单独声明的，不在 SCALAR_FIELDS 里，但必须带上——
# 下游要靠它定位到具体是哪一块。漏了它不会报错，只是结果里少个字段，
# 然后所有基于 chunk_id 的判定会静默失效。
OUTPUT_FIELDS = ["chunk_id"] + [name for name, _ in SCALAR_FIELDS]


# 表格单独收紧。表格里标点（竖线、逗号）密度高，而这类符号在
# BGE 的中文词表里没有对应项，会被拆成多个 token，
# 导致同样的字符数消耗的 token 多出不少。实测 1300 字仍会偶发被拒，
# 1100 字稳定通过，这里再留一点余量取 1000。
#
# 中英文各自的上限见 app/embeddings.py 的 estimate_char_limit()。
EMBED_MAX_CHARS_TABLE = 1000

# 逐条向量化仍然失败时，按这些比例依次缩短文本重试。
# 长度估算靠的是经验系数，边界上总有算不准的；短一点但有向量，
# 好过整块检索不到。
SHRINK_RATIOS = (1.0, 0.7, 0.5, 0.35)

# BM25 全文检索用的分词器。
#
# 不能留空用默认值：默认的 standard 分词器对中文完全无效，
# 中文查询一条都命中不了。而英文查询照样能中，所以只看英文会以为配置没问题。
#
# 实测（中文整词/部分词/人名/术语四类查询）：
#   standard     0/4      完全无效
#   whitespace   0/4      完全无效
#   jieba        4/4      可用
#   icu          4/4      可用，人名类分数更高
ANALYZER_TOKENIZER = "icu"

# BM25 稀疏向量字段名，以及对应的 BM25 函数名。
SPARSE_FIELD = "sparse"
BM25_FUNCTION = "bm25"

# RRF 融合参数。和原来 Qdrant 那套的 rrf_k 保持一致，便于对比。
RRF_K = 60

HTML_TAG = re.compile(r"<[^>]+>")


def escape_expr_value(value: str) -> str:
    """转义 Milvus 过滤表达式里的字符串字面量。

    item_id、parent_id 这类值由 f-string 拼进表达式，而它们来自 LLM 的
    工具参数——模型吐出一个双引号就能把表达式拆坏，甚至改变过滤语义。
    Milvus 表达式里反斜杠是转义符，所以把 \\ 和 " 都转掉。
    """

    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def strip_table_html(html: str) -> str:
    """把表格 HTML 拍成保留结构的纯文本。

    <table><tbody><tr><td> 这类标签不携带语义，却要占掉大量 token。
    单元格之间用空格分隔、行之间换行，表结构和内容都还在。

    分隔符用空格而不是竖线：竖线在 BGE 词表里会被拆成多个 token，
    几百个竖线叠加起来足以让请求超过 512 token 上限被拒。
    """

    text = re.sub(r"</t[dh]>", "  ", html)
    text = re.sub(r"</tr>", "\n", text)
    text = HTML_TAG.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def to_embed_text(chunk: dict) -> str:
    """算出用来生成向量的文本。

    与存进 Milvus 的 text 有两处不同：

    1. 表格剥掉 HTML 标签。表格的 text 是「表题 + HTML」，
       标签占了大部分长度却不表达任何语义。
    2. 按中英文比例估算安全长度并截断。纯中文和纯英文的
       「字符 / token」比例差三倍，用同一个上限会误伤其中一边。
    """

    text = str(chunk.get("text", "") or "")

    if chunk.get("block_type") == "table" and chunk.get("html"):
        text = f"{chunk.get('caption', '')}\n{strip_table_html(chunk['html'])}".strip()
        # 表格不能用固定上限（早期写死 1000，结果中文表格和标点密集的表格
        # 大量被拒）。走估算函数，再和表格专用上限取更保守的那个。
        limit = min(EMBED_MAX_CHARS_TABLE, estimate_char_limit(text))
    else:
        limit = estimate_char_limit(text)

    if len(text) > limit:
        text = text[:limit]

    return text or " "


def to_query_text(question: str) -> str:
    """算出用来生成查询向量的文本。

    文档侧有完整的收缩重试保护，查询侧原本是裸调用——用户贴一段长文
    当问题，embedding 接口直接 400，整个检索挂掉。这不是偶发：
    embedding 模型有 512 token 硬上限，超了每次都拒，重试三次也一样。

    截断而不是抛错。问句的后半段通常是补充说明，丢掉比整条挂掉好。
    """

    text = " ".join(str(question or "").split())
    if not text:
        return " "

    limit = estimate_char_limit(text)
    if len(text) > limit:
        logger.info("查询过长（%d 字），截断到 %d 字再向量化。", len(text), limit)
        text = text[:limit]

    return text or " "


@dataclass
class RetrievedChunk:
    """检索结果。字段和 qdrant_store.RetrievedChunk 保持一致，便于将来切换。"""

    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """一次检索的结果，连同「够不够置信」的判断。

    confident=False 时，调用方应当明确告诉用户「语料里没找到相关内容」，
    而不是把 chunks 里那些弱相关的段落端上去让模型硬答。那正是幻觉的
    来源——模型手里有东西，就会想办法用上。
    """

    chunks: list[RetrievedChunk]
    top_score: float | None
    confident: bool
    reranked: bool
    reason: str = ""


def parent_context_window(parent_text: str, child_text: str, max_chars: int) -> str:
    """从父块文本里截一段以命中子块为中心的窗口。

    子块命中说明答案在这附近，窗口对准它，前因后果都带一点。
    父块比预算短就原样返回；定位不到子块（文本归一化有差异时可能
    发生）就退回首截。省略号标记截断位置，让模型知道上下文不完整。
    """

    if len(parent_text) <= max_chars:
        return parent_text

    anchor = (child_text or "").strip()[:100]
    pos = parent_text.find(anchor) if anchor else -1
    if pos == -1:
        return parent_text[:max_chars] + "……"

    start = max(0, pos - max(0, (max_chars - 200) // 2))
    end = min(len(parent_text), start + max_chars)
    if pos + len(anchor) > end:
        # 锚点太靠后（理论上不会，start 以 pos 为中心）——保命中段。
        start = max(0, pos + len(anchor) - max_chars)
        end = min(len(parent_text), start + max_chars)

    prefix = "……" if start > 0 else ""
    suffix = "……" if end < len(parent_text) else ""
    return f"{prefix}{parent_text[start:end]}{suffix}"


def expand_chunks_to_parents(
    chunks: list[RetrievedChunk],
    parent_lookup,
    max_chars: int,
) -> list[RetrievedChunk]:
    """把命中子块替换成其父块的上下文窗口（small-to-big）。

    parent_lookup 是 parent_id -> 父块 dict | None 的查询函数
    （生产里传 ParentStore.get，测试里可以传假字典）。

    同一父块下的多个命中只保留排名最高的那个——它们共享同一段父块
    文本，重复放只会撑大 prompt。父块查不到（数据不一致）就原样
    保留子块，不让扩展失败影响可用结果。
    """

    expanded: list[RetrievedChunk] = []
    seen_parents: set[str] = set()

    for chunk in chunks:
        parent_id = str(chunk.metadata.get("parent_id") or "")
        if parent_id:
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            parent = parent_lookup(parent_id)
            if not parent:
                expanded.append(chunk)
                continue

            text = parent_context_window(
                str(parent.get("text") or ""), chunk.text, max_chars
            )
            metadata = dict(chunk.metadata)
            metadata["parent_expanded"] = True
            expanded.append(RetrievedChunk(score=chunk.score, text=text, metadata=metadata))
        else:
            expanded.append(chunk)

    return expanded


class MilvusStore:
    """Milvus 的读写封装。"""

    def __init__(
        self,
        settings: Settings | None = None,
        collection: str | None = None,
    ):
        self.settings = settings or get_settings()
        # 允许指定集合名，重建时用得上：读旧集合、写新集合。
        self.collection = collection or self.settings.milvus_collection
        self.embeddings = build_embeddings(self.settings)
        # 父块查找表：懒加载 parents.json（约 9MB），首次命中后才读。
        self.parent_store = ParentStore()

        token = self.settings.milvus_token or None
        self.client = MilvusClient(uri=self.settings.milvus_uri, token=token)

        # 向量化失败被跳过的块，以及缩短后才通过的块，供调用方汇总。
        self.skipped: list[str] = []
        self.shrunk: list[tuple[str, float]] = []

    def has_collection(self) -> bool:
        return self.client.has_collection(self.collection)

    def count(self) -> int:
        if not self.has_collection():
            return 0
        stats = self.client.get_collection_stats(self.collection)
        return int(stats.get("row_count", 0))

    def drop(self) -> None:
        if self.has_collection():
            self.client.drop_collection(self.collection)

    def ensure_collection(self, recreate: bool = False) -> None:
        """建集合与索引。已存在时默认不动。"""

        if recreate:
            self.drop()

        if self.has_collection():
            return

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=MAX_ID)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

        for name, max_length in SCALAR_FIELDS:
            if name == "is_reference":
                schema.add_field(name, DataType.BOOL)
            elif name in ("page", "page_end"):
                schema.add_field(name, DataType.INT32)
            elif name == "text":
                # text 字段要开分析器，BM25 才能从它生成稀疏向量。
                # 这个属性只能在建集合时设，事后改不了——
                # 所以带 BM25 的集合必须重建，不能给老集合补。
                schema.add_field(
                    name,
                    DataType.VARCHAR,
                    max_length=max_length,
                    enable_analyzer=True,
                    analyzer_params={"tokenizer": ANALYZER_TOKENIZER},
                )
            else:
                schema.add_field(name, DataType.VARCHAR, max_length=max_length)

        # 稀疏向量字段：内容由下面的 BM25 函数自动生成，不用我们填。
        schema.add_field(SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)

        schema.add_function(
            Function(
                name=BM25_FUNCTION,
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=[SPARSE_FIELD],
                params={},
            )
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        index_params.add_index(
            field_name=SPARSE_FIELD,
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
        )
        logger.info(
            "已创建 Milvus 集合：%s（全文检索分词器 %s）", self.collection, ANALYZER_TOKENIZER
        )

    def build_row(self, chunk: dict, vector: list[float]) -> dict:
        """把一条子块转成 Milvus 的行。"""

        def text_field(name: str, limit: int) -> str:
            value = str(chunk.get(name, "") or "")
            if len(value) > limit:
                logger.warning(
                    "字段 %s 超长被截断：%s 字符 -> %s", name, len(value), limit
                )
                return value[:limit]
            return value

        return {
            "chunk_id": text_field("chunk_id", MAX_ID),
            "vector": vector,
            "item_id": text_field("item_id", 16),
            "title": text_field("title", MAX_TITLE),
            "reader": text_field("reader", MAX_SHORT),
            "section": text_field("section", MAX_SECTION),
            "text": text_field("text", MAX_TEXT),
            "html": text_field("html", MAX_HTML),
            "parent_id": text_field("parent_id", MAX_ID),
            "block_type": text_field("block_type", 32),
            "label": text_field("label", MAX_SHORT),
            "image_path": text_field("image_path", MAX_PATH),
            "is_reference": bool(chunk.get("is_reference", False)),
            "page": int(chunk.get("page", 0) or 0),
            "page_end": int(chunk.get("page_end", 0) or 0),
        }

    def _embed_individually(self, batch: list[dict]) -> list[tuple[dict, list[float]]]:
        """逐条向量化，失败的依次缩短再试。

        批量接口是「全有或全无」：只要有一条文本不被接受，整批都报错。
        降级逐条能把真正有问题的挑出来，不让它拖垮其余几十条。

        仍然失败的话，按比例缩短文本再试。长度估算依赖的是经验系数，
        边界上总有算不准的；与其整块丢掉没有向量，不如用短一点但可检索的。

        缩短过的事实会记在 self.shrunk 里，便于事后看是哪一类文本。
        """

        pairs: list[tuple[dict, list[float]]] = []

        for chunk in batch:
            chunk_id = chunk.get("chunk_id", "?")
            text = to_embed_text(chunk)
            last_error: Exception | None = None

            for ratio in SHRINK_RATIOS:
                candidate = text[: max(1, int(len(text) * ratio))]
                try:
                    pairs.append((chunk, self.embeddings.embed_query(candidate)))
                    if ratio < 1.0:
                        self.shrunk.append((chunk_id, ratio))
                        logger.info("块 %s 缩短到 %s%% 后才通过向量化", chunk_id, int(ratio * 100))
                    break
                except Exception as exc:
                    last_error = exc

            else:
                self.skipped.append(chunk_id)
                logger.warning("块 %s 向量化失败，跳过：%s", chunk_id, last_error)

        return pairs

    def insert_chunks(self, chunks: list[dict], batch_size: int = 32) -> int:
        """批量写入子块。

        向量用批量接口计算：逐条调用在万级数据上要跑几个小时，
        批量能压到十几分钟。批量大小取 32，兼顾速度和单次失败的影响面。
        """

        if not chunks:
            return 0

        self.ensure_collection()
        written = 0
        total = len(chunks)

        for start in range(0, total, batch_size):
            batch = chunks[start : start + batch_size]

            try:
                vectors = self.embeddings.embed_documents(
                    [to_embed_text(c) for c in batch]
                )
                pairs = list(zip(batch, vectors))
            except Exception as exc:
                logger.warning("批量向量化失败（%s 条），降级逐条处理：%s", len(batch), exc)
                pairs = self._embed_individually(batch)

            if not pairs:
                continue

            rows = [self.build_row(chunk, vector) for chunk, vector in pairs]
            self.client.insert(collection_name=self.collection, data=rows)
            written += len(rows)

            if written % (batch_size * 10) < batch_size or written >= total:
                logger.info("已写入 %s/%s 条", written, total)

        self.client.flush(self.collection)
        return written

    def search(
        self,
        question: str,
        top_k: int | None = None,
        *,
        item_id: str | None = None,
        block_type: str | None = None,
        include_reference: bool = True,
    ) -> list[RetrievedChunk]:
        """向量检索子块。"""

        if not self.has_collection():
            return []

        limit = top_k or self.settings.retrieval_top_k
        vector = self.embeddings.embed_query(to_query_text(question))
        filter_expr = self.build_filter_expr(
            item_id=item_id,
            block_type=block_type,
            include_reference=include_reference,
        )

        results = self.client.search(
            collection_name=self.collection,
            data=[vector],
            # 必须显式指定搜哪个向量字段。集合里有两个（dense 的 vector 和
            # BM25 的 sparse），不指定会报 "multiple anns_fields exist"。
            anns_field="vector",
            limit=limit,
            output_fields=OUTPUT_FIELDS,
            filter=filter_expr,
        )

        chunks = []
        for hit in results[0] if results else []:
            entity = hit.get("entity", {}) or {}
            chunks.append(
                RetrievedChunk(
                    score=float(hit.get("distance", 0.0)),
                    text=entity.get("text", ""),
                    metadata=entity,
                )
            )
        return chunks

    def build_filter_expr(
        self,
        *,
        item_id: str | None = None,
        block_type: str | None = None,
        include_reference: bool = True,
    ) -> str | None:
        """构造过滤表达式，纯向量与混合检索两个分支共用。

        条件值来自 LLM 的工具参数，必须先转义再拼接（见 escape_expr_value）。
        """

        conditions = []
        if item_id:
            conditions.append(f'item_id == "{escape_expr_value(item_id)}"')
        if block_type:
            conditions.append(f'block_type == "{escape_expr_value(block_type)}"')
        if not include_reference:
            conditions.append("is_reference == false")
        return " and ".join(conditions) if conditions else None

    def resolve_query_pair(
        self,
        question: str,
        translate: bool = True,
    ) -> tuple[str, str] | None:
        """决定两条检索分支各自用什么查询文本。

        返回 (向量分支的查询, BM25 分支的查询)。
        返回 None 表示 BM25 用不上，调用方应退回纯向量。

        中文查询时两个分支用的是**不同的文本**：
          · 向量分支用中文原文——向量本身跨语言，中文查询能找到英文段落
          · BM25 分支用翻译后的英文——BM25 只认词面，中文词匹配不上英文文档

        实测（30 条跨语言样本，精确命中率）：
          中文向量 + 英文 BM25            90.0%

        为什么 BM25 分支不拼接中文原文（icu 分词器明明支持中文）：
        2026-09-21 做过完整诊断（scripts/diagnose_zh_bm25.py，135 条）。
        拼接确实能把 zh_fact 的混合层 recall 从 0.84 提到 1.00（中文词
        找回中文文献），但重排器会给「中文查询 × 中文文献」打高分——
        跨语言查询的正确英文文献被挤出 top-5，cross_lingual 在生产
        路径（重排+门槛）上从 0.983 掉到 0.817。净收益为负，已用数据
        否决。zh_fact 在生产路径的真实损失只有 4pp（0.96，重排器已经
        在补偿 BM25 的跨语言盲区），不值得为它牺牲跨语言。

        三层对照数据（混合层/重排层/生产层）见 dev-notes。
        """

        if not translate or not needs_translation(question):
            return question, question

        translation = translate_to_english(question, settings=self.settings)

        if not translation.translated:
            logger.info("查询未翻译成功（%s），BM25 分支跳过。", translation.reason)
            return None

        logger.info("查询已翻译：%s → %s", question, translation.english)
        return question, translation.english

    def hybrid_search(
        self,
        question: str,
        top_k: int | None = None,
        *,
        recall_limit: int = 20,
        item_id: str | None = None,
        block_type: str | None = None,
        include_reference: bool = True,
        translate: bool = True,
    ) -> list[RetrievedChunk]:
        """向量检索 + BM25 全文检索，RRF 融合后返回。

        两个分支各自召回 recall_limit 条，融合后取 top_k。
        召回数要比最终结果大，否则融合没有意义——两边的结果需要足够重叠
        才能体现各自的排序价值。

        只用 RRF，不做加权融合。原因是两条分支的分数尺度不可比：向量是
        COSINE 相似度，挤在 0.5~0.9 的窄区间；BM25 是词频得分，值域 0~20+。
        归一化后加权，会让分数尺度高的那一路单方面决定最终结果。

        300 条测试集上的实测（判分 240 条）：

            纯向量           83.3%
            RRF              93.8%   ← 采用
            加权 0.7:0.3     85.4%
            加权 0.5:0.5     84.2%
            加权 0.3:0.7     83.8%

        加权全线不如 RRF。而且权重给得越大，中文文献被英文 BM25 挤得越狠：
        0.5:0.5 和 0.3:0.7 都把中文题打到 24%，纯向量是 92%。详见
        docs/eval-set.md。

        translate:
          中文查询是否先翻译成英文再走 BM25。关掉它主要为了做对照实验。
        """

        if not self.has_collection():
            return []

        pair = self.resolve_query_pair(question, translate=translate)
        if pair is None:
            # 翻译没成功，BM25 用中文查询在英文语料上只会帮倒忙，退回纯向量。
            return self.search(
                question,
                top_k,
                item_id=item_id,
                block_type=block_type,
                include_reference=include_reference,
            )

        dense_query, sparse_query = pair

        limit = top_k or self.settings.retrieval_top_k
        expr = self.build_filter_expr(
            item_id=item_id,
            block_type=block_type,
            include_reference=include_reference,
        )

        dense_request = AnnSearchRequest(
            data=[self.embeddings.embed_query(to_query_text(dense_query))],
            anns_field="vector",
            param={"metric_type": "COSINE"},
            limit=recall_limit,
            expr=expr,
        )
        sparse_request = AnnSearchRequest(
            data=[sparse_query],
            anns_field=SPARSE_FIELD,
            param={"metric_type": "BM25"},
            limit=recall_limit,
            expr=expr,
        )

        ranker = RRFRanker(k=RRF_K)

        results = self.client.hybrid_search(
            collection_name=self.collection,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=limit,
            output_fields=OUTPUT_FIELDS,
        )

        chunks = []
        for hit in results[0] if results else []:
            entity = hit.get("entity", {}) or {}
            chunks.append(
                RetrievedChunk(
                    score=float(hit.get("distance", 0.0)),
                    text=entity.get("text", ""),
                    metadata=entity,
                )
            )
        return chunks

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        *,
        item_id: str | None = None,
        block_type: str | None = None,
        include_reference: bool = True,
        translate: bool = True,
        rerank: bool | None = None,
        recall_limit: int = 20,
        min_score: float | None = None,
        expand_parents: bool = False,
    ) -> RetrievalResult:
        """检索，并判断这批结果够不够回答这个问题。

        先召回一批，整批重排，再看重排后的最高分过不过门槛。

        为什么用重排分而不是检索分做判断：双塔余弦在跨语言场景下挤在
        0.5~0.6，该命中的（0.59）和语料外的（0.55）完全重叠，切不开；
        RRF 融合分只是名次的函数（第 2 名恒为 0.01639），原理上不携带
        相关性信息。只有重排分能画出一条线——实测门槛取 0.8 时，
        40 条语料外的问题全部拦住，155 条该命中的留住 88.4%。

        重排关掉或调用失败时不判置信度（confident 恒为 True），退回
        原来的行为。重排是锦上添花，不该因为它挂了就全线拒答。

        expand_parents: 命中子块后扩展成父块上下文（small-to-big）。
        排序与置信度判断都基于子块，扩展只发生在返回之前，改变的是
        给模型看的内容宽度，不影响召回、融合与判分。
        """

        limit = top_k or self.settings.retrieval_top_k
        use_rerank = self.settings.rerank_enabled if rerank is None else rerank
        threshold = self.settings.rerank_min_score if min_score is None else min_score

        # 重排要有得挑，所以融合阶段多返回一些，重排完再截回 top_k。
        candidate_count = max(limit, recall_limit) if use_rerank else limit

        chunks = self.hybrid_search(
            question,
            top_k=candidate_count,
            recall_limit=recall_limit,
            item_id=item_id,
            block_type=block_type,
            include_reference=include_reference,
            translate=translate,
        )

        if not chunks:
            return RetrievalResult([], None, False, False, "检索没有返回任何结果。")

        if not use_rerank:
            result = RetrievalResult(
                chunks[:limit], None, True, False, "未启用重排，不做置信度判断。"
            )
        else:
            ranked = rerank_documents(
                question, [chunk.text for chunk in chunks], settings=self.settings
            )
            if ranked is None:
                result = RetrievalResult(
                    chunks[:limit],
                    None,
                    True,
                    False,
                    "重排调用失败，退回原始顺序，不做置信度判断。",
                )
            else:
                reordered = [chunks[index] for index, _ in ranked]
                top_score = ranked[0][1]
                confident = top_score >= threshold
                result = RetrievalResult(
                    chunks=reordered[:limit],
                    top_score=top_score,
                    confident=confident,
                    reranked=True,
                    reason=(
                        f"重排最高分 {top_score:.4f}，"
                        + ("达到" if confident else "低于")
                        + f"门槛 {threshold}。"
                    ),
                )

        if expand_parents and result.chunks:
            try:
                result.chunks = expand_chunks_to_parents(
                    result.chunks,
                    self.parent_store.get,
                    self.settings.parent_expand_max_chars,
                )
            except Exception as exc:
                # 扩展是锦上添花，数据出问题时不挡检索结果。
                logger.warning("父块上下文扩展失败，返回原始子块：%s", exc)

        return result

    def query_all(
        self,
        output_fields: list[str] | None = None,
        batch_size: int = 300,
    ) -> list[dict]:
        """把集合里的数据整批读出来。

        用于重建集合时搬运数据。向量字段也能读出来，
        所以重建不用重新调 embedding 接口。

        用流式迭代器而不是带 offset 的 query：
        每次查询返回的数据量有上限，一次性拉 8000 多条带 1024 维向量的记录
        会直接报 "query results exceed the limit size"。
        迭代器由服务端分批吐，不受这个限制。

        batch_size 也不能给太大——每条记录除了向量还有几个文本字段，
        300 条一批大约 1.5 MB，比较稳。
        """

        if not self.has_collection():
            return []

        fields = output_fields or [
            name for name, _ in SCALAR_FIELDS
        ] + ["vector"]

        rows: list[dict] = []
        iterator = self.client.query_iterator(
            collection_name=self.collection,
            batch_size=batch_size,
            filter='chunk_id != ""',
            output_fields=fields,
        )

        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                rows.extend(batch)
        finally:
            iterator.close()

        return rows

    def get_by_parent(self, parent_id: str, limit: int = 50) -> list[dict]:
        """取某个父块下的全部子块，按页序返回。"""

        if not self.has_collection():
            return []

        rows = self.client.query(
            collection_name=self.collection,
            filter=f'parent_id == "{escape_expr_value(parent_id)}"',
            output_fields=["chunk_id", "text", "page", "block_type", "label"],
            limit=limit,
        )
        return sorted(rows, key=lambda r: (r.get("page", 0), r.get("chunk_id", "")))


class ParentStore:
    """父块的内存查找表。

    父块不参与向量检索，只在子块命中后用来补全上下文，
    因此存在本地 JSON 里，按 id 直接查，不走 Milvus。
    """

    def __init__(self, path: Path | None = None):
        self.path = path or PARENT_STORE_FILE
        self._parents: dict[str, dict] = {}
        self.loaded = False
        self._mtime_ns: int | None = None

    def load(self) -> int:
        if not self.path.exists():
            self._parents = {}
            self.loaded = True
            self._mtime_ns = None
            return 0

        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self._parents = {item["chunk_id"]: item for item in data}
        self.loaded = True
        self._mtime_ns = self.path.stat().st_mtime_ns
        return len(self._parents)

    def _ensure_fresh(self) -> None:
        current = self.path.stat().st_mtime_ns if self.path.exists() else None
        if not self.loaded or current != self._mtime_ns:
            self.load()

    def get(self, parent_id: str) -> dict | None:
        self._ensure_fresh()
        return self._parents.get(parent_id)

    def get_many(self, parent_ids: list[str]) -> list[dict]:
        self._ensure_fresh()
        seen = set()
        result = []
        for parent_id in parent_ids:
            if parent_id in seen:
                continue
            seen.add(parent_id)
            item = self._parents.get(parent_id)
            if item:
                result.append(item)
        return result

    def __len__(self) -> int:
        self._ensure_fresh()
        return len(self._parents)
