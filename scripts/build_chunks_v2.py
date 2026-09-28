import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from app.embeddings import estimate_char_limit


# 把规范化块切成父子结构的检索单元。
#
# 输入：data/processed/mineru_blocks/{item_id}.jsonl（parse_mineru.py 的产出）
# 输出：data/processed/chunks_v2.jsonl
#
# 父子切分的目的：
# - 子块小，检索准；命中后返回它所属的父块，上下文完整。
# - 表格、公式、图片是原子块，永远不切。
# - 文本按句子边界打包，不按字符数硬切。

BLOCK_DIR = PROJECT_DIR / "data" / "processed" / "mineru_blocks"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "chunks_v2.jsonl"
METADATA_DIR = PROJECT_DIR / "data" / "metadata"

INDEX_FILES = [
    METADATA_DIR / "literature_index_auto.csv",
    METADATA_DIR / "literature_index.csv",
]

# 子块的长度上限不写死。
#
# 它取决于 embedding 模型的 token 上限（bge-large-zh-v1.5 是 512），
# 而中文一个字约 1 token、英文一个字符约 0.36 token，差近三倍。
# 用同一个字符数上限会误伤其中一边，所以按内容的中文占比实时估算，
# 见 app/embeddings.py 的 estimate_char_limit()。
#
# 含义：中文子块切到 480 字左右，英文子块切到 1300 字左右。
# 切分和建库共用同一套判断，保证子块不会被 embedding 二次截断。

# 父块上限。超过就按子块边界拆成多个父块，
# 否则一个长章节（比如 Results）会变成几万字的巨型父块。
PARENT_MAX_CHARS = 3500

# 只有这些类型能作为子块内容的切分单位；其余一律视为原子块。
TEXT_TYPES = {"text", "ref_text", "page_footnote"}

# 原子块：本身完整，切开就废。
ATOMIC_TYPES = {"table", "figure", "equation"}

# 章节标题之前的块归到这个父块。
FRONT_MATTER = "（前置内容）"

# 参考文献整段不参与语义检索，打标记让下游能过滤。
REFERENCE_SECTION = "references"

SENTENCE_SPLIT = re.compile(r"(?<=[。！？；])|(?<=[.!?])(?=\s+[A-Z0-9(\[])")

SECTION_LABEL = re.compile(r"^(abstract|introduction|results?|discussion|conclusions?|"
                           r"methods?|materials|references|acknowledg|appendix)",
                           re.IGNORECASE)


def load_index() -> dict[str, dict]:
    """读索引表，建立 item_id -> 元数据 的映射。"""

    mapping: dict[str, dict] = {}
    for path in INDEX_FILES:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                item_id = (row.get("item_id") or "").strip()
                if item_id:
                    mapping[item_id] = row
        if mapping:
            break
    return mapping


def split_sentences(text: str) -> list[str]:
    """按句末标点切句。

    这是整个脚本的关键：原来的切分按字符数硬切，
    实测 92.6% 的块结尾不是句末标点，说明是从句子中间断的。
    """

    text = " ".join(str(text).split())
    if not text:
        return []

    parts = [p.strip() for p in SENTENCE_SPLIT.split(text)]
    return [p for p in parts if p]


def read_blocks(item_id: str) -> list[dict]:
    path = BLOCK_DIR / f"{item_id}.jsonl"
    if not path.exists():
        return []

    blocks = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                blocks.append(json.loads(line))
    return blocks


def group_by_section(blocks: list[dict]) -> list[tuple[str, list[dict]]]:
    """按章节标题把块分组。

    返回 [(章节名, 该章节的块列表), ...]。
    MinerU 只标注了 level 1/2，层级是平的，所以这里就按遇到的标题切分，
    不做嵌套。
    """

    groups: list[tuple[str, list[dict]]] = []
    current_title = FRONT_MATTER
    current: list[dict] = []

    for block in blocks:
        if block.get("block_type") == "section_title":
            groups.append((current_title, current))
            current_title = block.get("text", "").strip() or FRONT_MATTER
            current = []
            continue

        if block.get("block_type") == "doc_title":
            # 标题本身当元数据用，不放进正文。
            continue

        current.append(block)

    groups.append((current_title, current))
    return [(title, items) for title, items in groups if items]


def is_reference_section(section: str, blocks: list[dict]) -> bool:
    if REFERENCE_SECTION in section.lower():
        return True
    refs = sum(1 for b in blocks if b.get("block_type") == "ref_text")
    return refs > len(blocks) * 0.6


def build_children(section: str, blocks: list[dict], item_id: str, start: int) -> list[dict]:
    """把一个章节的块切成子块。"""

    children: list[dict] = []
    text_run: list[dict] = []
    counter = start

    def flush_text_run() -> None:
        nonlocal counter

        if not text_run:
            return

        sentences: list[tuple[str, int]] = []
        for block in text_run:
            for sentence in split_sentences(block.get("text", "")):
                sentences.append((sentence, block.get("page", 0)))

        buffer: list[str] = []
        pages: list[int] = []

        def emit(text: str, page_list: list[int]) -> None:
            nonlocal counter
            counter += 1
            children.append(
                {
                    "chunk_id": f"{item_id}_c{counter:04d}",
                    "chunk_type": "child",
                    "item_id": item_id,
                    "section": section,
                    "page": page_list[0] if page_list else 0,
                    "page_end": page_list[-1] if page_list else 0,
                    "block_type": "text",
                    "text": text.strip(),
                }
            )

        def flush_buffer() -> None:
            nonlocal buffer, pages
            if buffer:
                emit(" ".join(buffer), pages)
            buffer, pages = [], []

        for sentence, page in sentences:
            # 单句本身就超限时，buffer 塞不进去、也 flush 不掉，
            # 只能按上限硬切这一句。这是唯一的例外——其余都在句边界切。
            sentence_limit = estimate_char_limit(sentence)
            if len(sentence) > sentence_limit:
                flush_buffer()
                for start in range(0, len(sentence), sentence_limit):
                    emit(sentence[start : start + sentence_limit], [page])
                continue

            if buffer:
                # 阈值按「加入这一句之后」的内容算，所以会随中文占比浮动。
                candidate = " ".join(buffer + [sentence])
                if len(candidate) > estimate_char_limit(candidate):
                    flush_buffer()

            buffer.append(sentence)
            pages.append(page)

        flush_buffer()
        text_run.clear()

    for block in blocks:
        block_type = block.get("block_type", "")

        if block_type in ATOMIC_TYPES:
            flush_text_run()
            counter += 1
            child = {
                "chunk_id": f"{item_id}_c{counter:04d}",
                "chunk_type": "child",
                "item_id": item_id,
                "section": section,
                "page": block.get("page", 0),
                "page_end": block.get("page", 0),
                "block_type": block_type,
                "text": block.get("text", ""),
            }
            for key in ("html", "latex", "image_path", "label", "caption", "image_size"):
                if block.get(key):
                    child[key] = block[key]
            children.append(child)
            continue

        if block_type in TEXT_TYPES:
            text_run.append(block)
            continue

    flush_text_run()
    return children


def make_parents(
    item_id: str,
    section: str,
    children: list[dict],
    meta: dict,
    start: int,
) -> list[dict]:
    """按子块边界把章节内容打包成父块。

    章节太长时拆成多个父块，保证父块本身也是一个可读的上下文单元。
    """

    parents: list[dict] = []
    buffer: list[dict] = []
    buffer_len = 0
    counter = start

    def flush() -> None:
        nonlocal counter, buffer, buffer_len

        if not buffer:
            return
        counter += 1
        parents.append(
            {
                "chunk_id": f"{item_id}_p{counter:04d}",
                "chunk_type": "parent",
                "item_id": item_id,
                "section": section,
                "title": meta.get("title", ""),
                "page": buffer[0].get("page", 0),
                "page_end": buffer[-1].get("page_end", buffer[-1].get("page", 0)),
                "child_ids": [c["chunk_id"] for c in buffer],
                "text": "\n\n".join(c.get("text", "") for c in buffer),
            }
        )
        buffer, buffer_len = [], 0

    for child in children:
        text = child.get("text", "")
        if buffer_len + len(text) > PARENT_MAX_CHARS and buffer:
            flush()
        buffer.append(child)
        buffer_len += len(text) + 2

    flush()

    # 父块回填到子块，检索命中子块后能直接找到父块。
    for parent in parents:
        for child_id in parent["child_ids"]:
            for child in children:
                if child["chunk_id"] == child_id:
                    child["parent_id"] = parent["chunk_id"]
                    break

    return parents


def process_item(item_id: str, meta: dict) -> tuple[list[dict], list[dict]]:
    blocks = read_blocks(item_id)
    if not blocks:
        return [], []

    groups = group_by_section(blocks)
    all_parents: list[dict] = []
    all_children: list[dict] = []
    child_counter = 0
    parent_counter = 0

    for section, section_blocks in groups:
        children = build_children(section, section_blocks, item_id, child_counter)
        if not children:
            continue
        child_counter += len(children)

        parents = make_parents(item_id, section, children, meta, parent_counter)
        parent_counter += len(parents)

        is_ref = is_reference_section(section, section_blocks)
        for parent in parents:
            parent["is_reference"] = is_ref
        for child in children:
            child["is_reference"] = is_ref

        all_children.extend(children)
        all_parents.extend(parents)

    # 文献级元数据，子块父块都带一份，方便检索时过滤和展示来源。
    for chunk in all_parents + all_children:
        chunk["title"] = meta.get("title", "")
        chunk["reader"] = meta.get("reader", "")
        chunk["doi"] = meta.get("doi", "")

    return all_parents, all_children


def main() -> int:
    index = load_index()
    files = sorted(BLOCK_DIR.rglob("*.jsonl"))
    if not files:
        print(f"没有找到块文件：{BLOCK_DIR}")
        print("请先运行 scripts/parse_mineru.py")
        return 1

    print(f"块文件目录：{BLOCK_DIR}")
    print(f"输出文件  ：{OUTPUT_FILE}")
    print("子块上限  ：按中英文比例动态估算（中文约 480 字 / 英文约 1300 字）")
    print(f"父块上限 {PARENT_MAX_CHARS} 字")
    print("=" * 64)

    total_parents = total_children = 0
    written = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for path in files:
            item_id = path.stem
            meta = index.get(item_id, {})

            try:
                parents, children = process_item(item_id, meta)
            except Exception as exc:
                print(f"[{item_id}] 切分失败：{type(exc).__name__} {exc}")
                continue

            if not parents and not children:
                print(f"[{item_id}] 没有块，跳过")
                continue

            for chunk in parents + children:
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                written += 1

            total_parents += len(parents)
            total_children += len(children)

            sentences = sum(
                1 for c in children if c.get("block_type") == "text"
            )
            print(
                f"[{item_id}] 父块 {len(parents):>3}  子块 {len(children):>4}"
                f"（文本 {sentences}）"
            )

    print("=" * 64)
    print(f"文档 {len(files)} 篇   父块 {total_parents}   子块 {total_children}")
    print(f"共写出 {written} 条")

    if not OUTPUT_FILE.exists():
        return 1

    # 质量自检：子块长度分布、断句比例。
    check_quality()

    return 0


def check_quality() -> None:
    """对照旧版切分，确认断句问题确实改善了。"""

    parents = children = 0
    text_children: list[int] = []
    mid_cut = 0

    endings = "。！？.!?：:；;）)\"'”"

    with OUTPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("chunk_type") == "parent":
                parents += 1
                continue

            children += 1
            if chunk.get("block_type") != "text":
                continue
            text = chunk.get("text", "")
            text_children.append(len(text))
            if text and not text.rstrip().endswith(tuple(endings)):
                mid_cut += 1

    if not text_children:
        return

    text_children.sort()
    median = text_children[len(text_children) // 2]

    print()
    print("质量自检（仅统计文本子块）：")
    print(f"  文本子块 {len(text_children)} 个")
    print(f"  长度 最小 {text_children[0]} / 中位 {median} / 最大 {text_children[-1]}")
    print(f"  结尾不是句末标点（疑似断句）：{mid_cut}/{len(text_children)}"
          f" = {mid_cut/len(text_children):.1%}")
    print(f"  旧版切分该指标为 92.6%，越低说明按句子边界切得越好")


if __name__ == "__main__":
    raise SystemExit(main())
