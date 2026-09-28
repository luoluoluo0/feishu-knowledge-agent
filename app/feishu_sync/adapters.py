from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any, Iterable

from pypdf import PdfReader

from .models import NormalizedBlock, NormalizedDocument


HEADING_KEYS = {f"heading{i}" for i in range(1, 10)}
TEXT_KEYS = {"text", "bullet", "ordered", "code", "quote", "todo", "callout"}


def _collect_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_collect_text(item) for item in value)
    if not isinstance(value, dict):
        return ""
    text_run = value.get("text_run")
    if isinstance(text_run, dict):
        return str(text_run.get("content") or "")
    equation = value.get("equation")
    if isinstance(equation, dict):
        return str(equation.get("content") or "")
    if "elements" in value:
        return _collect_text(value.get("elements"))
    return "".join(_collect_text(item) for item in value.values())


def normalize_docx_blocks(
    *,
    document_id: str,
    item_id: str,
    title: str,
    source_url: str,
    blocks: Iterable[dict[str, Any]],
) -> NormalizedDocument:
    normalized: list[NormalizedBlock] = []
    raw_blocks = list(blocks)
    for index, block in enumerate(raw_blocks):
        block_id = str(block.get("block_id") or f"block-{index + 1}")
        payload_key = next(
            (key for key in (*HEADING_KEYS, *TEXT_KEYS) if isinstance(block.get(key), dict)),
            "",
        )
        if payload_key in HEADING_KEYS:
            text = _collect_text(block[payload_key]).strip()
            if text:
                normalized.append(NormalizedBlock(block_id, "heading", text=text))
            continue
        if payload_key in TEXT_KEYS:
            text = _collect_text(block[payload_key]).strip()
            if text:
                prefix = "• " if payload_key == "bullet" else ""
                normalized.append(NormalizedBlock(block_id, "text", text=prefix + text))
            continue

        if isinstance(block.get("table"), dict):
            text = _collect_text(block.get("table")).strip()
            normalized.append(
                NormalizedBlock(
                    block_id,
                    "table",
                    text=text or "[表格]",
                    html=f"<table><tr><td>{html.escape(text or '[表格]')}</td></tr></table>",
                )
            )
            continue
        if any(key in block for key in ("image", "file", "media")):
            normalized.append(NormalizedBlock(block_id, "text", text="[图片或附件]"))

    if not normalized:
        normalized.append(NormalizedBlock("empty", "text", text="[空文档]"))
    canonical = json.dumps(
        [block.as_ingestion_block() for block in normalized],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return NormalizedDocument(
        document_id=document_id,
        item_id=item_id,
        title=title,
        source_type="docx",
        source_url=source_url,
        content_hash=hashlib.sha256(canonical).hexdigest(),
        blocks=tuple(normalized),
    )


def normalize_pdf(
    *,
    document_id: str,
    item_id: str,
    title: str,
    source_url: str,
    path: Path,
    content_hash: str,
) -> NormalizedDocument:
    reader = PdfReader(str(path))
    extracted = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            extracted.append(
                NormalizedBlock(
                    block_id=f"page-{index}",
                    type="text",
                    text=text,
                    page=index,
                    page_end=index,
                )
            )
    blocks = tuple(extracted)
    if not blocks:
        raise ValueError("PDF 没有可提取文本，请安装 MinerU 或检查是否为扫描件")
    return NormalizedDocument(
        document_id=document_id,
        item_id=item_id,
        title=title,
        source_type="pdf",
        source_url=source_url,
        content_hash=content_hash,
        blocks=blocks,
        local_path=path,
    )
