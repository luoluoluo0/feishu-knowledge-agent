from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NormalizedBlock:
    block_id: str
    type: str
    text: str = ""
    html: str = ""
    section: str = ""
    page: int = 0
    page_end: int = 0
    image_path: str = ""
    label: str = ""

    def as_ingestion_block(self) -> dict[str, Any]:
        block_type = self.type
        if block_type == "heading":
            block_type = "section_title"
        elif block_type not in {
            "doc_title",
            "section_title",
            "text",
            "ref_text",
            "page_footnote",
            "table",
            "figure",
            "equation",
        }:
            block_type = "text"
        payload = {
            "block_id": self.block_id,
            "block_type": block_type,
            "text": self.text,
            "page": self.page,
        }
        for key in ("html", "image_path", "label"):
            value = getattr(self, key)
            if value:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class NormalizedDocument:
    document_id: str
    item_id: str
    title: str
    source_type: str
    source_url: str
    content_hash: str
    blocks: tuple[NormalizedBlock, ...] = field(default_factory=tuple)
    local_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["local_path"] = str(self.local_path) if self.local_path else None
        return payload


@dataclass(frozen=True)
class SyncJob:
    id: int
    source_token: str
    job_type: str
    target_hash: str
    status: str
    attempts: int
    lease_owner: str = ""
    lease_until: str = ""
