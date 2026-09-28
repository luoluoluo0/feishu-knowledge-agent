import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.qdrant_store import QdrantStore, RetrievedChunk


logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_DIR / "data" / "processed" / "chunks.jsonl"


@dataclass
class BM25Document:
    """BM25 索引中的一条资料块。"""

    chunk: dict[str, Any]
    tokens: list[str]
    term_freq: Counter


@dataclass
class HybridCandidate:
    """BM25 和 Qdrant 融合后的中间结果。"""

    key: str
    rrf_score: float
    text: str
    metadata: dict[str, Any]
    bm25_rank: int | None = None
    qdrant_rank: int | None = None
    bm25_score: float | None = None
    qdrant_score: float | None = None


def is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def tokenize(text: str) -> list[str]:
    """简单中文/英文分词，用于 BM25。

    这里不用 jieba，是为了减少项目依赖。中文用单字和双字组合，能覆盖人名、
    标题关键词、编号等精确检索场景。
    """
    text = (text or "").lower()
    ascii_words = re.findall(r"[a-z0-9]+", text)
    cjk_chars = [char for char in text if is_cjk(char)]
    cjk_bigrams = [
        cjk_chars[index] + cjk_chars[index + 1]
        for index in range(len(cjk_chars) - 1)
    ]
    return ascii_words + cjk_chars + cjk_bigrams


def normalize_item_id(item_id: str | None) -> str | None:
    if item_id is None:
        return None

    item_id = item_id.strip()
    if item_id.isdigit():
        return item_id.zfill(3)
    return item_id


def build_bm25_text(chunk: dict[str, Any]) -> str:
    """拼出 BM25 检索文本。

    标题、编号、整理者重复几次，相当于给这些结构化字段更高权重。
    literature_card 只用结构化字段，避免飞书长文档正文污染卡片检索。
    """
    title = chunk.get("title", "")
    item_id = chunk.get("item_id", "")
    reader = chunk.get("reader", "")
    doi = chunk.get("doi", "")
    keywords = chunk.get("keywords", "")
    content_type = chunk.get("content_type", "")
    source_file = chunk.get("source_file", "")
    text = "" if content_type == "literature_card" else chunk.get("text", "")

    return "\n".join(
        [
            title,
            title,
            title,
            item_id,
            item_id,
            reader,
            reader,
            reader,
            doi,
            keywords,
            content_type,
            source_file,
            text,
        ]
    )


def clean_card_text(text: str, content_type: str | None) -> str:
    if content_type == "literature_card":
        return text.split("飞书笔记内容：", 1)[0]
    return text


def load_chunks(path: Path = CHUNKS_FILE) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"没有找到分块文件：{path}")

    chunks = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


class BM25Index:
    """内存版 BM25 索引。

    服务启动时构建一次，后续查询直接复用。
    """

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: list[BM25Document] = []
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.avg_doc_len = 0.0
        self._build(chunks)

    def _build(self, chunks: list[dict[str, Any]]) -> None:
        total_length = 0

        for chunk in chunks:
            tokens = tokenize(build_bm25_text(chunk))
            term_freq = Counter(tokens)
            self.documents.append(
                BM25Document(
                    chunk=chunk,
                    tokens=tokens,
                    term_freq=term_freq,
                )
            )
            total_length += len(tokens)

            for token in term_freq:
                self.doc_freq[token] += 1

        if self.documents:
            self.avg_doc_len = total_length / len(self.documents)

    def idf(self, token: str) -> float:
        doc_count = len(self.documents)
        freq = self.doc_freq.get(token, 0)
        return math.log((doc_count - freq + 0.5) / (freq + 0.5) + 1)

    def score_document(self, query_tokens: list[str], document: BM25Document) -> float:
        score = 0.0
        doc_len = len(document.tokens)

        if doc_len == 0 or self.avg_doc_len == 0:
            return 0.0

        for token in query_tokens:
            tf = document.term_freq.get(token, 0)
            if tf == 0:
                continue

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += self.idf(token) * numerator / denominator

        return score

    def search(
        self,
        query: str,
        *,
        top_k: int,
        content_type: str | None = None,
        item_id: str | None = None,
    ) -> list[tuple[float, BM25Document]]:
        query_tokens = tokenize(query)
        item_id = normalize_item_id(item_id)
        scored_results = []

        for document in self.documents:
            chunk = document.chunk

            if content_type and chunk.get("content_type") != content_type:
                continue
            if item_id and chunk.get("item_id") != item_id:
                continue

            score = self.score_document(query_tokens, document)
            if score > 0:
                scored_results.append((score, document))

        scored_results.sort(key=lambda item: item[0], reverse=True)
        return scored_results[:top_k]


def result_key(metadata: dict[str, Any], *, dedupe_by_item: bool) -> str:
    if dedupe_by_item:
        return f"item:{metadata.get('item_id', '')}"
    return f"chunk:{metadata.get('parent_chunk_id') or metadata.get('chunk_id', '')}"


def rrf_score(rank: int, rrf_k: int) -> float:
    return 1 / (rrf_k + rank)


class HybridRetriever:
    """BM25 + Qdrant + RRF 的混合检索器。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        chunks_file: Path = CHUNKS_FILE,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ):
        self.settings = settings or get_settings()
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.qdrant_store = QdrantStore(self.settings)
        self.bm25_index = BM25Index(load_chunks(chunks_file))

    def merge(
        self,
        *,
        bm25_results: list[tuple[float, BM25Document]],
        qdrant_results: list[RetrievedChunk],
        dedupe_by_item: bool,
    ) -> list[HybridCandidate]:
        merged: dict[str, HybridCandidate] = {}

        for rank, (score, document) in enumerate(bm25_results, start=1):
            chunk = document.chunk
            metadata = {key: value for key, value in chunk.items() if key != "text"}
            content_type = metadata.get("content_type", "")
            key = result_key(metadata, dedupe_by_item=dedupe_by_item)

            merged[key] = HybridCandidate(
                key=key,
                rrf_score=rrf_score(rank, self.rrf_k),
                bm25_rank=rank,
                qdrant_rank=None,
                bm25_score=score,
                qdrant_score=None,
                text=clean_card_text(chunk.get("text", ""), content_type),
                metadata=metadata,
            )

        for rank, chunk in enumerate(qdrant_results, start=1):
            metadata = dict(chunk.metadata)
            content_type = metadata.get("content_type", "")
            key = result_key(metadata, dedupe_by_item=dedupe_by_item)
            score = rrf_score(rank, self.rrf_k)

            if key not in merged:
                merged[key] = HybridCandidate(
                    key=key,
                    rrf_score=score,
                    bm25_rank=None,
                    qdrant_rank=rank,
                    bm25_score=None,
                    qdrant_score=chunk.score,
                    text=clean_card_text(chunk.text, content_type),
                    metadata=metadata,
                )
                continue

            merged[key].rrf_score += score
            merged[key].qdrant_rank = rank
            merged[key].qdrant_score = chunk.score

            if not merged[key].text.strip():
                merged[key].text = clean_card_text(chunk.text, content_type)

        return sorted(merged.values(), key=lambda item: item.rrf_score, reverse=True)

    def to_retrieved_chunks(
        self,
        candidates: list[HybridCandidate],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        chunks = []

        for candidate in candidates[:top_k]:
            metadata = dict(candidate.metadata)
            metadata["retrieval_method"] = "hybrid_rrf"
            metadata["rrf_score"] = candidate.rrf_score
            metadata["bm25_rank"] = candidate.bm25_rank
            metadata["qdrant_rank"] = candidate.qdrant_rank
            metadata["bm25_score"] = candidate.bm25_score
            metadata["qdrant_score"] = candidate.qdrant_score

            chunks.append(
                RetrievedChunk(
                    score=candidate.rrf_score,
                    text=candidate.text,
                    metadata=metadata,
                )
            )

        return chunks

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        content_type: str | None = None,
        item_id: str | None = None,
        dedupe_by_item: bool = False,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.retrieval_top_k
        item_id = normalize_item_id(item_id)

        bm25_results = self.bm25_index.search(
            query,
            top_k=self.candidate_k,
            content_type=content_type,
            item_id=item_id,
        )

        try:
            qdrant_results = self.qdrant_store.search(
                query,
                top_k=self.candidate_k,
                content_type=content_type,
                item_id=item_id,
            )
        except Exception as exc:
            # Qdrant 挂了就只用 BM25 那一路，但不能无声降级——
            # 检索质量为什么掉下来了，日志里得能查到原因。
            logger.warning("Qdrant 检索失败，仅用 BM25 结果：%s", exc)
            qdrant_results = []

        candidates = self.merge(
            bm25_results=bm25_results,
            qdrant_results=qdrant_results,
            dedupe_by_item=dedupe_by_item,
        )
        return self.to_retrieved_chunks(candidates, top_k=top_k)

    def search_ppt(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return self.search(query, top_k=top_k, content_type="slide")

    def search_literature_card(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return self.search(
            query,
            top_k=top_k,
            content_type="literature_card",
            dedupe_by_item=True,
        )
