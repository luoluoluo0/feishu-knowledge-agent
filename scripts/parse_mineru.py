import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import pymupdf

from app.config import Settings, get_settings


# 把 MinerU 的 middle_json 转成项目自己的规范化块格式。
#
# MinerU 输出的是「版面块」——标题、正文、公式、表格、图片各自成块，
# 带页码和归一化坐标。这个脚本只做规范化和必要的合并，不做语义切分；
# 切分由后续脚本负责，因为切分参数需要反复调，而裁图很慢，
# 两者分开可以避免每次调参都重跑一遍裁图。
#
# 输入：
#   data/processed/mineru_json/{文件名}.json
#   data/raw/papers/{文件名}.pdf
# 输出：
#   data/processed/mineru_blocks/{item_id}.jsonl
#   data/processed/figures/{item_id}_fig{序号}.png


MINERU_JSON_DIR = PROJECT_DIR / "data" / "processed" / "mineru_json"
PAPER_DIR = PROJECT_DIR / "data" / "raw" / "papers"
BLOCK_DIR = PROJECT_DIR / "data" / "processed" / "mineru_blocks"
FIGURE_DIR = PROJECT_DIR / "data" / "processed" / "figures"
METADATA_DIR = PROJECT_DIR / "data" / "metadata"

INDEX_FILES = [
    METADATA_DIR / "literature_index_auto.csv",
    METADATA_DIR / "literature_index.csv",
]

# 裁图分辨率。200 够组会展示；调高会显著增加体积。
FIGURE_DPI = 200

# 版面噪音：页眉、页脚、页码。这些块不含正文信息，直接丢弃。
# page_footnote 保留，它通常是作者单位、基金号这类有用信息。
NOISE_TYPES = {"header", "footer", "page_number"}

# 小于这个页面占比的图块当装饰图标丢弃（实测 CC 图标约占 0.2%）。
MIN_FIGURE_AREA = 0.02

# 没有图注的图块也丢弃，它们通常是图标、logo，检索价值低。
REQUIRE_FIGURE_CAPTION = True

LABEL_PATTERN = re.compile(r"^\s*(Figure|Fig\.?|Table)\s*(\d{1,3})", re.IGNORECASE)


def load_index() -> dict[str, str]:
    """建立「PDF 文件名 -> item_id」的映射。"""

    mapping: dict[str, str] = {}
    for path in INDEX_FILES:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                paper_file = (row.get("paper_file") or "").strip()
                item_id = (row.get("item_id") or "").strip()
                if paper_file and item_id:
                    mapping[paper_file] = item_id
        if mapping:
            break
    return mapping


def inline_text(content: Any) -> str:
    """把块的 content 拍平成一段纯文本。

    注意 MinerU 的 content 有三种形态：

    - 字符串：只有 equation 块是这样，内容是 LaTeX
    - 列表：元素是 {type, content, styles?}
    - **列表里嵌套列表**：子元素的 content 本身又是一个列表

    第三种容易漏。之前用 str() 处理，结果把 Python 字面量当正文写了进去：

        [{'type': 'text', 'content': 'https://doi.org/10.1007/...'}]. 66. Sadri,

    所以这里递归处理，嵌套多深都能拍平。
    """

    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return ""

    pieces = []
    for item in content:
        if isinstance(item, dict):
            pieces.append(inline_text(item.get("content", "")))
        else:
            pieces.append(str(item))
    return "".join(pieces)


def pick_part(content: Any, part_type: str) -> dict | None:
    """从块的 content 列表里取出指定类型的子部件（表格体、图注等）。"""

    if not isinstance(content, list):
        return None
    for item in content:
        if isinstance(item, dict) and item.get("type") == part_type:
            return item
    return None


def extract_label(text: str) -> str:
    """从「Figure 3. xxx」里取出「Figure 3」这样的标注。"""

    match = LABEL_PATTERN.match(text or "")
    if not match:
        return ""
    return f"{match.group(1).rstrip('.')} {match.group(2)}"


def normalize_block(
    block: dict,
    *,
    item_id: str,
    page_number: int,
) -> dict | None:
    """把一个 MinerU 版面块规范化成项目自己的块。返回 None 表示丢弃。"""

    block_type = block.get("type", "")
    if block_type in NOISE_TYPES:
        return None

    base = {
        "item_id": item_id,
        "page": page_number,
        "block_index": block.get("index"),
        "bbox": block.get("bbox"),
        "order": block.get("order"),
        "continues_prev": bool(block.get("continues_prev")),
    }

    if block_type in ("text", "ref_text", "page_footnote", "doc_title", "paragraph_title"):
        text = inline_text(block.get("content")).strip()
        if not text:
            return None
        block_kind = "section_title" if block_type == "paragraph_title" else block_type
        if block_type == "doc_title":
            block_kind = "doc_title"
        return {
            **base,
            "block_type": block_kind,
            "level": block.get("level"),
            "text": text,
        }

    if block_type == "equation":
        latex = inline_text(block.get("content")).strip()
        if not latex:
            return None
        return {**base, "block_type": "equation", "latex": latex, "text": latex}

    if block_type == "table":
        body = pick_part(block.get("content"), "table_body")
        caption = pick_part(block.get("content"), "table_caption")
        html = str(body.get("content", "")).strip() if body else ""
        caption_text = inline_text(caption.get("content")).strip() if caption else ""
        if not html:
            return None
        return {
            **base,
            "block_type": "table",
            "html": html,
            "caption": caption_text,
            "label": extract_label(caption_text),
            "text": "\n".join(part for part in (caption_text, html) if part),
        }

    if block_type == "image":
        body = pick_part(block.get("content"), "image_body")
        caption = pick_part(block.get("content"), "image_caption")
        caption_text = inline_text(caption.get("content")).strip() if caption else ""
        return {
            **base,
            "block_type": "figure",
            "bbox": (body or block).get("bbox"),
            "caption": caption_text,
            "label": extract_label(caption_text),
            "area": block_area((body or block).get("bbox")),
            "text": caption_text,
        }

    return None


def block_area(bbox: Any) -> float:
    """归一化 bbox 的面积占比。"""

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def merge_cross_page(blocks: list[dict]) -> list[dict]:
    """把跨页断开的正文块接回去。

    MinerU 用 continues_prev 标记「本块是上一页某块的续接」。
    不合并的话，一个完整段落会被切成两半。
    """

    merged: list[dict] = []

    for block in blocks:
        if block.get("continues_prev") and merged:
            previous = merged[-1]
            same_kind = previous.get("block_type") == block.get("block_type")
            if same_kind and block.get("block_type") in ("text", "ref_text"):
                previous["text"] = f"{previous['text']} {block['text']}".strip()
                previous.setdefault("merged_pages", [previous["page"]])
                previous["merged_pages"].append(block["page"])
                continue

        block.pop("continues_prev", None)
        merged.append(block)

    return merged


def crop_figure(
    document: Any,
    page_index: int,
    bbox: list[float],
    page_size: dict,
    out_path: Path,
) -> tuple[int, int]:
    """按归一化 bbox 从原 PDF 裁出图片。"""

    width = page_size.get("width_pt") or document[page_index].rect.width
    height = page_size.get("height_pt") or document[page_index].rect.height

    clip = pymupdf.Rect(
        bbox[0] * width,
        bbox[1] * height,
        bbox[2] * width,
        bbox[3] * height,
    )
    pix = document[page_index].get_pixmap(clip=clip, dpi=FIGURE_DPI)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out_path))
    return pix.width, pix.height


def parse_one(json_path: Path, pdf_path: Path, item_id: str) -> dict:
    """解析一篇文档，写出块文件和图片。"""

    data = json.loads(json_path.read_text(encoding="utf-8"))
    page_sizes = {
        page["page_idx"]: page
        for page in data.get("extensions", {})
        .get("docvortex_layout", {})
        .get("pages", [])
    }

    # 先按页内顺序收集，保留原文顺序。
    raw_blocks: list[dict] = []
    order = 0
    for page in data.get("pages", []):
        page_index = page.get("page_idx", 0)
        for block in page.get("blocks", []):
            order += 1
            normalized = normalize_block(
                block, item_id=item_id, page_number=page_index + 1
            )
            if normalized is None:
                continue
            normalized["order"] = order
            raw_blocks.append(normalized)

    blocks = merge_cross_page(raw_blocks)

    # 裁图，同时过滤掉装饰性图标。
    document = pymupdf.open(str(pdf_path))
    kept: list[dict] = []
    figure_no = 0
    dropped_figures = 0

    for block in blocks:
        if block.get("block_type") != "figure":
            kept.append(block)
            continue

        if block.get("area", 0) < MIN_FIGURE_AREA:
            dropped_figures += 1
            continue
        if REQUIRE_FIGURE_CAPTION and not block.get("caption"):
            dropped_figures += 1
            continue

        bbox = block.get("bbox")
        if not bbox:
            dropped_figures += 1
            continue

        figure_no += 1
        name = f"{item_id}_fig{figure_no:02d}.png"
        out_path = FIGURE_DIR / name
        try:
            width, height = crop_figure(
                document,
                block["page"] - 1,
                bbox,
                page_sizes.get(block["page"] - 1, {}),
                out_path,
            )
        except Exception as exc:
            print(f"    裁图失败 第{block['page']}页：{exc}")
            dropped_figures += 1
            continue

        block["image_path"] = str(out_path.relative_to(PROJECT_DIR)).replace("\\", "/")
        block["image_size"] = f"{width}x{height}"
        kept.append(block)

    document.close()

    BLOCK_DIR.mkdir(parents=True, exist_ok=True)
    out_file = BLOCK_DIR / f"{item_id}.jsonl"
    with out_file.open("w", encoding="utf-8") as file:
        for block in kept:
            file.write(json.dumps(block, ensure_ascii=False) + "\n")

    stats: dict[str, int] = {}
    for block in kept:
        stats[block["block_type"]] = stats.get(block["block_type"], 0) + 1

    return {
        "item_id": item_id,
        "blocks": len(kept),
        "figures": stats.get("figure", 0),
        "dropped_figures": dropped_figures,
        "stats": stats,
        "out_file": out_file,
    }


def find_pdfs() -> Iterable[tuple[str, Path, Path]]:
    """遍历已有 MinerU 结果的文档，产出 (item_id, json, pdf)。"""

    index = load_index()
    # 递归扫描：MinerU 按目录批量解析时可能建子目录，不能只 glob 一层。
    for json_path in sorted(MINERU_JSON_DIR.rglob("*.json")):
        if json_path.name.startswith("_"):
            continue
        stem = json_path.stem
        pdf_path = PAPER_DIR / f"{stem}.pdf"
        if not pdf_path.exists():
            print(f"找不到 PDF，跳过：{stem}")
            continue

        item_id = index.get(f"{stem}.pdf", "")
        if not item_id:
            print(f"索引里查不到 item_id，跳过：{stem}")
            continue

        yield item_id, json_path, pdf_path


def main() -> int:
    settings: Settings = get_settings()
    print(f"MinerU 结果目录：{MINERU_JSON_DIR}")
    print(f"图片输出目录  ：{FIGURE_DIR}")
    print(f"裁图 DPI      ：{FIGURE_DPI}")
    print(f"Qdrant 集合   ：{settings.qdrant_collection}（本脚本不写库）")
    print()

    targets = list(find_pdfs())
    if not targets:
        print("没有找到可解析的 MinerU 结果。")
        return 1

    print(f"待解析 {len(targets)} 篇")
    print("=" * 64)

    total_blocks = total_figures = total_dropped = 0
    combined: dict[str, int] = {}

    for item_id, json_path, pdf_path in targets:
        try:
            result = parse_one(json_path, pdf_path, item_id)
        except Exception as exc:
            print(f"[{item_id}] 解析失败：{type(exc).__name__} {exc}")
            continue

        total_blocks += result["blocks"]
        total_figures += result["figures"]
        total_dropped += result["dropped_figures"]
        for key, value in result["stats"].items():
            combined[key] = combined.get(key, 0) + value

        detail = " ".join(f"{k}={v}" for k, v in sorted(result["stats"].items()))
        print(f"[{item_id}] {result['blocks']:>4} 块  {detail}")

    print("=" * 64)
    print(f"文档 {len(targets)} 篇   块 {total_blocks}")
    print(f"图片 {total_figures} 张（丢弃装饰性 {total_dropped} 张）")
    print()
    print("块类型合计：")
    for key, value in sorted(combined.items(), key=lambda x: -x[1]):
        print(f"  {key:<16} {value}")
    print()
    print(f"块文件：{BLOCK_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
