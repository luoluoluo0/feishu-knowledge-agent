from dataclasses import dataclass
from typing import Any

import requests

from app.config import Settings
from app.embeddings import build_embeddings


@dataclass
class RetrievedChunk:
    score: float
    text: str
    metadata: dict[str, Any]


class QdrantStore:
    """负责连接 Qdrant，并把用户问题检索成相关资料块。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = build_embeddings(settings)

    def build_filter(
        self,
        *,
        content_type: str | None = None,
        item_id: str | None = None,
        title: str | None = None,
    ) -> dict | None:
        """构造 Qdrant 的 metadata 过滤条件。

        现在先做精确过滤：
        - content_type: literature_card / paper_text / slide
        - item_id: 001 / 002 / ...
        - title: 完整文献标题
        """
        must_conditions = []

        if content_type:
            must_conditions.append(
                {
                    "key": "content_type",
                    "match": {"value": content_type},
                }
            )

        if item_id:
            must_conditions.append(
                {
                    "key": "item_id",
                    "match": {"value": item_id},
                }
            )

        if title:
            must_conditions.append(
                {
                    "key": "title",
                    "match": {"value": title},
                }
            )

        if not must_conditions:
            return None

        return {"must": must_conditions}

    def search(
        self,
        question: str,
        top_k: int | None = None,
        *,
        content_type: str | None = None,
        item_id: str | None = None,
        title: str | None = None,
    ) -> list[RetrievedChunk]:
        """向量检索。

        不传过滤条件时，就是全库相似度检索。
        传 content_type / item_id / title 时，就只在指定范围里检索。
        """
        query_vector = self.embeddings.embed_query(question)
        limit = top_k or self.settings.retrieval_top_k

        payload = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True,
        }
        qdrant_filter = self.build_filter(
            content_type=content_type,
            item_id=item_id,
            title=title,
        )
        if qdrant_filter:
            payload["filter"] = qdrant_filter

        response = requests.post(
            f"{self.settings.qdrant_url}/collections/{self.settings.qdrant_collection}/points/search",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        chunks = []
        for item in response.json().get("result", []):
            payload = item.get("payload", {})
            chunks.append(
                RetrievedChunk(
                    score=float(item.get("score", 0)),
                    text=payload.get("text", ""),
                    metadata={key: value for key, value in payload.items() if key != "text"},
                )
            )

        return chunks

    def search_by_item(self, question: str, item_id: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """只检索某一篇文献的资料。"""
        return self.search(question, top_k=top_k, item_id=item_id)
