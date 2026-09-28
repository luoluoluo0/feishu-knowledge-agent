import csv
import os
import re
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path
from zipfile import ZipFile


# 这个脚本的作用：
# 1. 读取飞书导出的 markdown 文献分享笔记
# 2. 按“每篇文献”拆成条目
# 3. 从每个条目里提取阅读者、DOI、PDF 附件名、PPT 附件名
# 4. 去本地 papers / ppts 文件夹匹配真实文件
# 5. 生成 literature_index_auto.csv，给后续分块和向量化使用

PROJECT_DIR = Path(__file__).resolve().parent.parent
NOTE_DIR = PROJECT_DIR / "data" / "raw" / "feishu_notes"
PAPER_DIR = PROJECT_DIR / "data" / "raw" / "papers"
PPT_DIR = PROJECT_DIR / "data" / "raw" / "ppts"
METADATA_DIR = PROJECT_DIR / "data" / "metadata"
OUTPUT_FILE = METADATA_DIR / "literature_index_auto.csv"
DEBUG_ATTACHMENTS_FILE = METADATA_DIR / "attachment_debug.csv"
OVERRIDES_FILE = METADATA_DIR / "literature_index_overrides.csv"


def normalize_text(text: str) -> str:
    """把字符串变成适合模糊匹配的形式：去掉空格、标点、大小写差异。"""
    return re.sub(r"[\s\W_]+", "", text.lower())


def get_item_id(number: str) -> str:
    """把 1、2、3 统一变成 001、002、003。"""
    value = int(number)
    if value <= 0:
        value = 1
    return str(value).zfill(3)


def find_note_file() -> Path:
    """找到飞书笔记文件。

    优先级：
    1. 环境变量 FEISHU_NOTE_FILE 指定的文件
    2. feishu_notes 里的 docx
    3. feishu_notes 里的 markdown

    docx 的段落结构通常比 markdown 导出更稳定，所以默认优先尝试 docx。
    """

    specified = os.getenv("FEISHU_NOTE_FILE", "").strip()
    if specified:
        path = Path(specified)
        if not path.is_absolute():
            path = NOTE_DIR / specified
        if not path.exists():
            raise FileNotFoundError(f"FEISHU_NOTE_FILE 指定的文件不存在：{path}")
        return path

    files = sorted(NOTE_DIR.glob("*.docx"))
    if files:
        return files[0]

    files = sorted(NOTE_DIR.glob("*.md"))
    if files:
        return files[0]

    raise FileNotFoundError(f"没有找到飞书 docx 或 markdown 文件：{NOTE_DIR}")


def read_docx_text(path: Path) -> str:
    """用标准库读取 docx 段落文本。

    docx 本质是 zip 包，正文在 word/document.xml 里。
    这里不依赖 python-docx，避免额外安装包。
    """

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []

    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))

    for paragraph in root.findall(".//w:p", namespace):
        texts = [
            text_node.text or ""
            for text_node in paragraph.findall(".//w:t", namespace)
        ]
        text = "".join(texts).strip()
        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


def read_note_text(path: Path) -> str:
    """读取飞书笔记文本，支持 markdown 和 docx。"""

    suffix = path.suffix.lower()
    if suffix == ".docx":
        return read_docx_text(path)
    if suffix == ".md":
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(
            "unknown",
            b"",
            0,
            1,
            f"无法识别 markdown 文件编码：{path}",
        )
    raise ValueError(f"暂不支持的飞书笔记格式：{path}")


def clean_title(title: str) -> str:
    """清理标题两侧的 markdown 加粗符号和多余空格。"""
    title = title.strip()
    title = re.sub(r"^\*+", "", title)
    title = re.sub(r"\*+$", "", title)
    title = title.strip()
    title = re.sub(r"\s+", " ", title)
    return title


def clean_text(text: str) -> str:
    """清理文本中的多余空格和空行，主要用于诊断预览。"""
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_literature_items(text: str) -> list[dict]:
    """按文献标题把飞书 markdown 拆成多个文献条目。

    优先识别带编号的标题，比如：
    - 1. 标题
    - ## 1. 标题
    - **1. 标题**
    - 1\\. 标题

    如果飞书导出时把编号丢了，就走 split_literature_items_without_number。
    """
    numbered_heading_pattern = re.compile(
        r"(?m)^\s*(?:[-*+]\s*)?(?:#{1,6}\s*)?(?:\*\*)?\s*"
        r"(\d{1,3})\s*(?:\\?\.|．|、|\)|）)\s*"
        r"(.+?)\s*(?:\*\*)?\s*$"
    )
    matches = list(numbered_heading_pattern.finditer(text))

    if not matches:
        return split_literature_items_without_number(text)

    items = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        number = match.group(1)
        title = clean_title(match.group(2))
        body = text[start:end].strip()

        if not title or title == "样例":
            continue

        if int(number) <= 0:
            number = str(len(items) + 1)

        items.append(
            {
                "item_id": get_item_id(number),
                "title": title,
                "body": body,
            }
        )

    # 如果编号重复，说明飞书导出格式有点怪，就按出现顺序重新编号。
    item_ids = [item["item_id"] for item in items]
    if len(item_ids) != len(set(item_ids)):
        for index, item in enumerate(items, start=1):
            item["item_id"] = get_item_id(str(index))

    return items


def is_noise_heading(title: str) -> bool:
    """过滤掉文档总标题、目录等不是真正文献条目的标题。"""
    title = clean_title(title)
    noise_titles = {
        "信息分化-文献阅读分享笔记",
        "信息分化文献阅读分享笔记",
        "目录",
        "阅读笔记",
    }

    if title in noise_titles:
        return True

    if len(title) <= 1:
        return True

    return False


def split_literature_items_without_number(text: str) -> list[dict]:
    """飞书 markdown 没有保留编号时，按标题出现顺序生成 001、002、003。"""
    heading_pattern = re.compile(r"(?m)^\s*(?:#{1,6}\s+|\*\*)(.+?)(?:\*\*)?\s*$")
    matches = []

    for match in heading_pattern.finditer(text):
        title = clean_title(match.group(1))
        if is_noise_heading(title):
            continue
        matches.append((match, title))

    items = []
    for index, (match, title) in enumerate(matches):
        start = match.end()
        end = matches[index + 1][0].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        items.append(
            {
                "item_id": get_item_id(str(index + 1)),
                "title": title,
                "body": body,
            }
        )

    return items


def extract_reader(body: str) -> str:
    """从每篇文献正文里提取阅读整理者。"""
    patterns = [
        r"阅读整理者[：:]\s*(.+)",
        r"阅读者[：:]\s*(.+)",
        r"分享人[：:]\s*(.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, body)
        if match:
            return match.group(1).strip()

    return ""


def extract_doi(body: str) -> str:
    """从正文中提取 DOI 链接；如果只有 10.xxxx/xxx，也补成 doi.org 链接。"""
    doi_url = re.search(r"https?://doi\.org/[^\s\)）]+", body)
    if doi_url:
        return doi_url.group(0).strip()

    doi_raw = re.search(r"\b10\.\d{4,9}/[^\s\)）]+", body)
    if doi_raw:
        return "https://doi.org/" + doi_raw.group(0).strip()

    return ""


def extract_keywords(title: str, body: str) -> str:
    """用一组简单关键词做粗标注，后面可以人工改得更细。"""
    words = []
    candidate_text = title + "\n" + body[:500]
    keyword_candidates = [
        "信息分化",
        "偏见同化",
        "同质性",
        "极化",
        "谣言",
        "社交网络",
        "推荐系统",
        "舆论",
        "信息茧房",
        "算法",
        "信任",
        "知识图谱",
        "因果",
        "机器学习",
    ]

    for word in keyword_candidates:
        if word in candidate_text and word not in words:
            words.append(word)

    return ";".join(words)


def normalize_filename_name(name: str) -> str:
    """把文件名去掉后缀后规范化，用于附件名和本地文件名的模糊匹配。"""
    return normalize_text(Path(name).stem)


def is_generic_attachment_name(name: str) -> bool:
    """判断附件名是不是过于泛化。

    这类名字只有在本地文件精确同名时才能匹配。
    如果继续用包含关系猜，很容易把 “组会.pptx” 错配成 “1025组会PPT.pptx”。
    """

    key = normalize_filename_name(name)
    generic_keys = {
        "ppt",
        "pptx",
        "组会",
        "组会ppt",
        "汇报",
        "汇报ppt",
        "论文",
        "全文",
        "阅读笔记",
    }

    if key in generic_keys:
        return True

    # 001.pptx、02.pptx 这种只有编号的附件名也不能随便模糊匹配。
    if key.isdigit():
        return True

    return False


def clean_attachment_name(name: str) -> str:
    """清理从 markdown 里提取出的附件名。

    飞书 markdown 会把 .、-、_、括号等符号转义成 \\.、\\-、\\_。
    本地真实文件名没有这些反斜杠，所以匹配前要删掉。
    """
    name = name.strip()
    name = re.sub(r"^[\-\*\+\s\[\(（]+", "", name)
    name = re.sub(r"[\]\)）\s]+$", "", name)
    name = name.replace("%20", " ")
    name = name.replace("\\", "")
    return name.strip()


def extract_attachment_files(body: str, suffixes: tuple[str, ...]) -> list[str]:
    """从一篇文献的 markdown 正文里提取 PDF/PPT 附件名。

    这里优先使用 .md 里写明的附件关系，而不是靠标题或日期猜。
    """
    # 先匹配 .pptx，再匹配 .ppt，避免 xxx.pptx 被截断成 xxx.ppt。
    suffixes = tuple(sorted(suffixes, key=len, reverse=True))
    extension_pattern = "|".join(re.escape(suffix.lstrip(".")) for suffix in suffixes)
    files: list[str] = []

    # 匹配 markdown 链接形式：[xxx.pptx](url)
    link_pattern = re.compile(
        rf"\[([^\]\n\r]+?\\?\.({extension_pattern}))\]\([^\)]*\)",
        re.IGNORECASE,
    )
    for match in link_pattern.finditer(body):
        files.append(clean_attachment_name(match.group(1)))

    # 匹配普通文本形式：xxx.pptx 或 xxx.pdf。
    plain_pattern = re.compile(
        rf"([^\n\r\[\]]{{1,160}}?\\?\.({extension_pattern}))",
        re.IGNORECASE,
    )
    for match in plain_pattern.finditer(body):
        filename = clean_attachment_name(match.group(1))
        if filename and filename not in files:
            files.append(filename)

    return files


def find_existing_by_attachment(folder: Path, attachment_file: str, suffixes: tuple[str, ...]) -> str:
    """根据 markdown 中提取出的附件名，去本地文件夹找真实文件。"""
    if not attachment_file:
        return ""

    direct_path = folder / attachment_file
    if direct_path.exists():
        return direct_path.name

    attachment_key = normalize_filename_name(attachment_file)
    if not attachment_key:
        return ""

    candidates = []
    for file in sorted(folder.iterdir()):
        if not file.is_file():
            continue
        if file.suffix.lower() not in suffixes:
            continue

        file_key = normalize_filename_name(file.name)
        candidates.append((file, file_key))

        # 规范化后完全一致，也属于确定匹配。
        if file_key == attachment_key:
            return file.name

    # 泛化附件名只允许精确匹配，精确找不到就留空。
    if is_generic_attachment_name(attachment_file):
        return ""

    # 足够具体的附件名才允许包含匹配，避免短词误匹配。
    if len(attachment_key) < 8:
        return ""

    for file, file_key in candidates:
        if attachment_key in file_key or file_key in attachment_key:
            return file.name

    return ""


def find_existing_by_attachments(folder: Path, attachment_files: list[str], suffixes: tuple[str, ...]) -> str:
    """一篇文献可能有多个 PDF 或 PPT，这里依次尝试匹配，找到第一个就返回。"""
    for attachment_file in attachment_files:
        matched = find_existing_by_attachment(folder, attachment_file, suffixes)
        if matched:
            return matched
    return ""


def natural_ppt_sort_key(filename: str):
    """让日期型 PPT 按真实日期排序，比如 2023.9.27 排在 2023.10.11 前面。"""
    match = re.search(r"(\d{4})[.\-_年](\d{1,2})[.\-_月](\d{1,2})", filename)
    if match:
        year, month, day = (int(value) for value in match.groups())
        return (0, year, month, day, filename)
    return (1, 9999, 99, 99, filename)


def list_ppt_files() -> list[str]:
    """列出本地 ppts 文件夹里的 PPT 文件，用于打印诊断和可选兜底匹配。"""
    if not PPT_DIR.exists():
        return []

    return sorted(
        [
            file.name
            for file in PPT_DIR.iterdir()
            if file.is_file() and file.suffix.lower() in {".ppt", ".pptx"}
        ],
        key=natural_ppt_sort_key,
    )


def fill_missing_ppt_by_order(rows: list[dict], ppt_files: list[str]) -> list[dict]:
    """按 PPT 文件顺序给缺失条目补 ppt_file。

    默认不启用，因为它有错绑风险。
    只有设置环境变量 AUTO_FILL_PPT_BY_ORDER=1 时才会使用。
    """
    used_ppts = {row.get("ppt_file", "") for row in rows if row.get("ppt_file")}
    remaining_ppts = [name for name in ppt_files if name not in used_ppts]

    if not remaining_ppts:
        return rows

    ppt_index = 0
    for row in rows:
        if row.get("ppt_file"):
            continue
        if ppt_index >= len(remaining_ppts):
            break

        row["ppt_file"] = remaining_ppts[ppt_index]
        row["status"] = get_status(row.get("paper_file", ""), row.get("ppt_file", ""))
        row["match_note"] = "ppt_filled_by_order"
        ppt_index += 1

    return rows


def load_manual_overrides() -> dict[str, dict]:
    """读取人工校正表。

    自动解析只能提取飞书笔记里明确写出来的信息。
    如果飞书里没有 reader / 分享人，或者标题、附件被粘连，就在
    literature_index_overrides.csv 里按 item_id 手动补。
    """

    if not OVERRIDES_FILE.exists():
        return {}

    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with OVERRIDES_FILE.open("r", encoding=encoding, newline="") as file:
                rows = list(csv.DictReader(file))
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise UnicodeDecodeError(
            "unknown",
            b"",
            0,
            1,
            f"无法识别人工校正表编码：{OVERRIDES_FILE}，最后错误：{last_error}",
        )

    overrides = {}
    for row in rows:
        item_id = (row.get("item_id") or "").strip()
        if not item_id:
            continue
        overrides[item_id] = {
            key: (value or "").strip()
            for key, value in row.items()
            if key != "item_id"
        }

    return overrides


def apply_manual_overrides(rows: list[dict]) -> list[dict]:
    """把人工校正表覆盖到自动生成的索引结果里。"""

    overrides = load_manual_overrides()
    if not overrides:
        return rows

    editable_fields = {
        "title",
        "reader",
        "doi",
        "paper_file",
        "ppt_file",
        "note_file",
        "keywords",
        "theme",
        "status",
        "match_note",
    }

    changed = 0
    for row in rows:
        item_id = row.get("item_id", "")
        override = overrides.get(item_id)
        if not override:
            continue

        row_changed = False
        for field in editable_fields:
            value = override.get(field, "")
            if value and value != (row.get(field, "") or ""):
                row[field] = value
                row_changed = True

        if row_changed:
            if not override.get("status", ""):
                row["status"] = get_status(row.get("paper_file", ""), row.get("ppt_file", ""))
            notes = [row.get("match_note", ""), "manual_override"]
            row["match_note"] = ";".join(note for note in notes if note)
            changed += 1

    print(f"人工校正覆盖：{changed} 条")
    return rows


def find_file_by_item_id(folder: Path, item_id: str, suffixes: tuple[str, ...]) -> str:
    """根据 001、002 这种编号匹配文件，比如 001_xxx.pdf 或 001-xxx.pdf。"""
    if not folder.exists():
        return ""

    for file in sorted(folder.iterdir()):
        if not file.is_file():
            continue
        if file.suffix.lower() not in suffixes:
            continue
        if re.match(rf"^{re.escape(item_id)}[\-_ ]?", file.name):
            return file.name

    return ""


def find_file_by_title(folder: Path, title: str, suffixes: tuple[str, ...]) -> str:
    """标题和文件名高度一致时，用包含关系匹配。"""
    if not folder.exists():
        return ""

    title_key = normalize_text(title)
    if not title_key:
        return ""

    for file in sorted(folder.iterdir()):
        if not file.is_file():
            continue
        if file.suffix.lower() not in suffixes:
            continue

        file_key = normalize_text(file.stem)
        if title_key and (title_key in file_key or file_key in title_key):
            return file.name

    return ""


def similarity(a: str, b: str) -> float:
    """计算两个字符串的相似度，用于标题和文件名不完全一致的情况。"""
    return SequenceMatcher(None, a, b).ratio()


def find_file_by_fuzzy_title(folder: Path, title: str, suffixes: tuple[str, ...]) -> str:
    """标题和文件名不完全一致时，用模糊相似度找最像的文件。"""
    if not folder.exists():
        return ""

    title_key = normalize_text(title)
    if not title_key:
        return ""

    best_file = ""
    best_score = 0.0

    for file in sorted(folder.iterdir()):
        if not file.is_file():
            continue
        if file.suffix.lower() not in suffixes:
            continue

        file_key = normalize_text(file.stem)
        score = similarity(title_key, file_key)

        if score > best_score:
            best_score = score
            best_file = file.name

    if best_score >= 0.45:
        return best_file

    return ""


def find_local_file(
    folder: Path,
    item_id: str,
    title: str,
    suffixes: tuple[str, ...],
    attachment_file: str = "",
    attachment_files: list[str] | None = None,
) -> str:
    """综合多种方式匹配本地文件。

    匹配优先级：
    1. markdown 里明确写出的附件名
    2. item_id 编号
    3. 标题包含关系
    4. 标题模糊匹配
    """
    by_attachments = find_existing_by_attachments(folder, attachment_files or [], suffixes)
    if by_attachments:
        return by_attachments

    by_attachment = find_existing_by_attachment(folder, attachment_file, suffixes)
    if by_attachment:
        return by_attachment

    by_id = find_file_by_item_id(folder, item_id, suffixes)
    if by_id:
        return by_id

    by_title = find_file_by_title(folder, title, suffixes)
    if by_title:
        return by_title

    return find_file_by_fuzzy_title(folder, title, suffixes)


def get_status(paper_file: str, ppt_file: str) -> str:
    """根据 PDF/PPT 是否匹配到，生成资料状态。"""
    if paper_file and ppt_file:
        return "ready"
    if paper_file and not ppt_file:
        return "missing_ppt"
    if ppt_file and not paper_file:
        return "missing_pdf"
    return "note_only"


def build_index():
    """主流程：解析飞书笔记，匹配本地附件，组装 metadata 行。"""
    note_file = find_note_file()
    print(f"使用飞书笔记：{note_file}")
    text = read_note_text(note_file)
    items = split_literature_items(text)

    if not items:
        print("没有识别到文献条目。请检查飞书笔记标题格式。")
        return []

    ppt_files = list_ppt_files()
    if ppt_files:
        print(f"检测到 PPT 文件数量：{len(ppt_files)}")
        print("前 10 个 PPT 文件：")
        for name in ppt_files[:10]:
            print("  ", name)
    else:
        print(f"没有在目录中检测到 PPT 文件：{PPT_DIR}")

    rows = []
    debug_rows = []

    for item in items:
        item_id = item["item_id"]
        title = item["title"]
        body = item["body"]

        # 从当前文献条目的正文里提取附件名，这是最可靠的匹配依据。
        paper_attachments = extract_attachment_files(body, (".pdf",))
        ppt_attachments = extract_attachment_files(body, (".ppt", ".pptx"))
        paper_attachment = paper_attachments[0] if paper_attachments else ""
        ppt_attachment = ppt_attachments[0] if ppt_attachments else ""

        paper_file = find_local_file(
            PAPER_DIR,
            item_id,
            title,
            (".pdf",),
            paper_attachment,
            paper_attachments,
        )
        ppt_file = find_local_file(
            PPT_DIR,
            item_id,
            title,
            (".ppt", ".pptx"),
            ppt_attachment,
            ppt_attachments,
        )

        rows.append(
            {
                "item_id": item_id,
                "title": title,
                "reader": extract_reader(body),
                "doi": extract_doi(body),
                "paper_file": paper_file,
                "ppt_file": ppt_file,
                "note_file": note_file.name,
                "keywords": extract_keywords(title, body),
                "theme": "信息分化",
                "status": get_status(paper_file, ppt_file),
                "match_note": "from_markdown_attachment" if ppt_file and ppt_attachments else "",
            }
        )

        # attachment_debug.csv 是排错用的，不参与后续向量化。
        debug_rows.append(
            {
                "item_id": item_id,
                "title": title,
                "paper_attachments_in_md": " | ".join(paper_attachments),
                "ppt_attachments_in_md": " | ".join(ppt_attachments),
                "matched_paper_file": paper_file,
                "matched_ppt_file": ppt_file,
                "body_preview": clean_text(body[:300]).replace("\n", " "),
            }
        )

    # 默认关闭顺序兜底匹配，避免 PPT 错绑。
    # 需要时可以在终端临时设置 AUTO_FILL_PPT_BY_ORDER=1 再运行。
    if os.getenv("AUTO_FILL_PPT_BY_ORDER", "").lower() in {"1", "true", "yes"}:
        rows = fill_missing_ppt_by_order(rows, ppt_files)

    rows = apply_manual_overrides(rows)
    save_attachment_debug(debug_rows)
    return rows


def save_attachment_debug(rows: list[dict]):
    """保存附件诊断表，用来检查 markdown 里提取到了什么附件、最终匹配到了什么文件。"""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id",
        "title",
        "paper_attachments_in_md",
        "ppt_attachments_in_md",
        "matched_paper_file",
        "matched_ppt_file",
        "body_preview",
    ]

    with DEBUG_ATTACHMENTS_FILE.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_csv(rows: list[dict]):
    """保存最终自动生成的 literature_index_auto.csv。"""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id",
        "title",
        "reader",
        "doi",
        "paper_file",
        "ppt_file",
        "note_file",
        "keywords",
        "theme",
        "status",
        "match_note",
    ]

    with OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = build_index()
    save_csv(rows)

    ready_count = sum(row["status"] == "ready" for row in rows)
    print(f"生成完成：{OUTPUT_FILE}")
    print(f"附件诊断：{DEBUG_ATTACHMENTS_FILE}")
    print(f"文献数量：{len(rows)}")
    print(f"资料齐全：{ready_count}")
    print("状态统计：")

    for status in ["ready", "missing_pdf", "missing_ppt", "note_only"]:
        count = sum(row["status"] == status for row in rows)
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
