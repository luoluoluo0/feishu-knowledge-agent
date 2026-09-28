"""单篇增量入库：新论文 PDF → 解析 → 切块 → Milvus 更新 → 质检 → 登记。

解决"新进一篇论文要全量重建"的问题：只动这一篇的向量、父块和索引行，
存量论文一行不碰。全量重建走 pipeline.py --mode full。

用法：
    python scripts/ingest_paper.py <pdf路径> --item-id 120 --title "标题" [--reader 姓名] [--force]

幂等：同一 PDF（SHA-256 相同）重复执行会自动跳过；文件变了（哈希不同）
或显式 --force 才走更新（先删该 item_id 的旧块再插新的）。

入库后跑 5 道质检，任一失败自动回滚 Milvus 与本地合并文件：

    QC1 子块数 > 0
    QC2 Milvus 按 item_id 查回数量 == 插入数量
    QC3 父块 child_ids 与子块 parent_id 双向一致
    QC4 抽样检索：论文标题检索 top3 必须命中本篇
    QC5 集合总行数 == 插入前 + 净增量（证明存量未被动过）
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.milvus_store import MilvusStore

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

MANIFEST_FILE = PROJECT_DIR / "data" / "processed" / "ingest_manifest.jsonl"
CHUNKS_FILE = PROJECT_DIR / "data" / "processed" / "chunks_v2.jsonl"
PARENTS_FILE = PROJECT_DIR / "data" / "processed" / "parents.json"
INDEX_CSV = PROJECT_DIR / "data" / "metadata" / "literature_index_auto.csv"
OVERRIDES_CSV = PROJECT_DIR / "data" / "metadata" / "literature_index_overrides.csv"

CSV_FIELDS = [
    "item_id", "title", "reader", "doi", "paper_file", "ppt_file",
    "note_file", "keywords", "theme", "status", "match_note",
]


# --------------------------------------------------------------------------
# 纯逻辑函数（tests/test_ingest.py 直接测试这些）


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, dict]:
    """读 manifest，按 item_id 归并（后写覆盖先写）。"""

    entries: dict[str, dict] = {}
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entries[entry["item_id"]] = entry
    return entries


def save_manifest(path: Path, entries: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for entry in sorted(entries.values(), key=lambda e: e["item_id"]):
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def needs_ingest(entry: dict | None, pdf_hash: str, force: bool) -> tuple[bool, str]:
    """判断要不要入库。返回 (需要入库, 原因)。"""

    if entry is None:
        return True, "新文献"
    if force:
        return True, "显式 --force"
    if entry.get("pdf_sha256") != pdf_hash:
        return True, "PDF 内容已变化（哈希不同）"
    return False, "同一 PDF 已入库（幂等跳过）"


def merge_parents_by_item(old_parents: list[dict], item_id: str, new_parents: list[dict]) -> list[dict]:
    """父块按 item_id 替换：剔除旧的，追加新的，其他论文不动。"""

    kept = [p for p in old_parents if p.get("item_id") != item_id]
    return kept + list(new_parents)


def merge_chunks_lines(old_lines: list[str], item_id: str, new_children: list[dict]) -> list[str]:
    """chunks_v2.jsonl 行级合并：剔除该 item_id 的旧行，追加新子块行。"""

    prefix = f'"item_id": "{item_id}"'
    kept = [line for line in old_lines if prefix not in line]
    kept.extend(json.dumps(child, ensure_ascii=False) for child in new_children)
    return kept


def upsert_csv_row(rows: list[dict], item_id: str, fields: dict) -> tuple[list[dict], str]:
    """CSV 行级 upsert。返回 (新行列表, 动作 update/append)。"""

    action = "update"
    found = False
    for row in rows:
        if row.get("item_id") == item_id:
            row.update({k: v for k, v in fields.items() if k != "item_id"})
            found = True
            break
    if not found:
        action = "append"
        row = {key: "" for key in CSV_FIELDS}
        row.update(fields)
        row["item_id"] = item_id
        rows.append(row)
    return rows, action


def qc_parent_child_consistent(parents: list[dict], children: list[dict]) -> tuple[bool, str]:
    """父块 child_ids 与子块 parent_id 双向一致才放行。"""

    child_ids = {c.get("chunk_id") for c in children}
    parent_ids = {p.get("chunk_id") for p in parents}

    for child in children:
        pid = child.get("parent_id")
        if pid not in parent_ids:
            return False, f"子块 {child.get('chunk_id')} 的 parent_id={pid} 在父块里找不到"
    for parent in parents:
        for cid in parent.get("child_ids", []):
            if cid not in child_ids:
                return False, f"父块 {parent.get('chunk_id')} 引用的子块 {cid} 不存在"
        if parent.get("item_id") != parents[0].get("item_id"):
            return False, "混入了其他论文的父块"
    if parent_ids & child_ids:
        return False, "父子 chunk_id 有交集，编号异常"
    return True, "一致"


def next_item_id(existing_ids: list[str]) -> str:
    """现有最大数字编号 + 1，三位补零。"""

    numbers = [int(x) for x in existing_ids if x.isdigit()]
    return f"{max(numbers, default=0) + 1:03d}"


# --------------------------------------------------------------------------
# 带副作用的流程步骤（惰性导入重依赖，保持模块可轻量测试）


def resolve_item_id(pdf: Path, csv_rows: list[dict], explicit: str | None) -> str:
    """编号解析顺序：显式参数 > CSV paper_file 匹配 > 最大编号 +1。"""

    if explicit:
        return explicit.strip().zfill(3) if explicit.strip().isdigit() else explicit.strip()

    for row in csv_rows:
        if row.get("paper_file") == pdf.name:
            return row["item_id"]

    return next_item_id([row.get("item_id", "") for row in csv_rows])


def run_mineru_parse(pdf: Path) -> Path:
    """调 mineru-kit 解析单个 PDF，返回 middle_json 路径。"""

    import run_mineru_batch

    mineru_cmd = run_mineru_batch.find_mineru()
    ok, message = run_mineru_batch.parse_one(mineru_cmd, pdf, timeout=600)
    if not ok:
        raise RuntimeError(f"MinerU 解析失败：{message}")

    json_path = run_mineru_batch.OUTPUT_DIR / f"{pdf.stem}.json"
    if not json_path.exists():
        raise RuntimeError(f"MinerU 未产出预期文件：{json_path}")
    return json_path


def parse_blocks(json_path: Path, pdf: Path, item_id: str) -> dict:
    """middle_json → 规范化块 jsonl（parse_mineru.parse_one 单篇复用）。"""

    import parse_mineru

    return parse_mineru.parse_one(json_path, pdf, item_id)


def chunk_item(item_id: str, meta: dict) -> tuple[list[dict], list[dict]]:
    """规范化块 → 父子块（build_chunks_v2.process_item 单篇复用）。"""

    import build_chunks_v2

    blocks = build_chunks_v2.read_blocks(item_id)
    if not blocks:
        raise RuntimeError(f"块文件为空：{item_id}.jsonl（parse 阶段可能失败）")
    return build_chunks_v2.process_item(item_id, meta)


def load_index_rows() -> list[dict]:
    with INDEX_CSV.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def save_index_rows(rows: list[dict]) -> None:
    with INDEX_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_override(item_id: str, fields: dict) -> None:
    """把增量条目写进 overrides，防止重跑 build_literature_index 时丢行。"""

    exists = OVERRIDES_CSV.exists()
    with OVERRIDES_CSV.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        row = {key: "" for key in CSV_FIELDS}
        row.update(fields)
        row["item_id"] = item_id
        row["match_note"] = "incremental_ingest"
        writer.writerow(row)


def delete_item_chunks(store: MilvusStore, item_id: str) -> int:
    """按 item_id 删除子块并等它真正生效，返回删除前行数。

    两个实测教训：
    1. MilvusClient.delete 的 deleted_count 不可靠（实测 filter 删除
       返回空 dict/0，但行确实被删了）——真实数量要用 query 数出来。
    2. 删除是异步生效的，不 flush 就插入会让新旧两份同主键块共存，
       检索行为变得不可预测（QC4 因此误报过）。
    """

    old_rows = store.client.query(
        collection_name=store.collection,
        filter=f'item_id == "{item_id}"',
        output_fields=["chunk_id"],
        limit=16384,
    )
    store.client.delete(collection_name=store.collection, filter=f'item_id == "{item_id}"')
    store.client.flush(store.collection)
    return len(old_rows)


def item_chunk_count(store: MilvusStore, item_id: str) -> int:
    rows = store.client.query(
        collection_name=store.collection,
        filter=f'item_id == "{item_id}"',
        output_fields=["chunk_id"],
        limit=16384,
    )
    return len(rows)


class IngestionError(RuntimeError):
    """入库失败。质检失败时已自动回滚，failures 是质检明细。"""

    def __init__(self, message: str, failures: list[str] | None = None):
        super().__init__(message)
        self.failures = failures or []


def extract_title_from_blocks(item_id: str) -> str:
    """未提供标题时，从解析块的 doc_title 提取论文真实标题。

    下载的论文 PDF 文件名常是出版商编号（如 1-s2.0-S0952197623003196-main），
    拿它当标题去检索必然命中不了本篇——QC4 拦下过这个真实案例。
    """

    import build_chunks_v2

    for block in build_chunks_v2.read_blocks(item_id):
        if block.get("block_type") == "doc_title":
            text = str(block.get("text", "")).strip()
            if text:
                return text
    return ""


def _run_ingestion_unlocked(
    pdf: Path,
    *,
    item_id: str | None = None,
    title: str = "",
    reader: str = "",
    theme: str = "",
    force: bool = False,
    log=print,
) -> dict:
    """单篇增量入库完整流程，CLI 与 API 共用。

    返回报告 dict（status=ingested 或 skipped）；任何失败抛 IngestionError，
    质检失败时 Milvus 与本地合并文件已自动回滚。
    """

    from app.config import get_settings
    from app.milvus_store import MilvusStore

    pdf = Path(pdf)
    if not pdf.exists():
        raise IngestionError(f"PDF 不存在：{pdf}")

    csv_rows = load_index_rows()
    item_id = resolve_item_id(pdf, csv_rows, item_id)
    existing_row = next(
        (row for row in csv_rows if row.get("item_id") == item_id), None
    )
    csv_title = (existing_row or {}).get("title", "")

    # 版本判断
    pdf_hash = sha256_file(pdf)
    manifest = load_manifest(MANIFEST_FILE)
    needed, reason = needs_ingest(manifest.get(item_id), pdf_hash, force)
    log(f"目标：item_id={item_id}  《{title or csv_title or pdf.stem}》")
    log(f"版本判断：{reason}")
    if not needed:
        return {"item_id": item_id, "status": "skipped", "reason": reason}

    settings = get_settings()
    store = MilvusStore(settings)
    store.ensure_collection()
    # 存量抽查样本：3 篇其他论文的行数快照。不能用 get_collection_stats
    # 的总行数做 QC——删除是最终一致的，已删行在 compaction 前仍被计入
    # （实测 8303 vs 实际 8227），精确断言必然误报。
    others = [row["item_id"] for row in csv_rows if row.get("item_id") != item_id]
    sample_ids = [sid for sid in others if sid][:3]
    sample_before = {sid: item_chunk_count(store, sid) for sid in sample_ids}
    log(f"集合统计约 {store.count()} 行（最终一致），存量抽查样本：{sample_ids}")

    log("[1/5] MinerU 解析…")
    json_path = run_mineru_parse(pdf)

    log("[2/5] 规范化块…")
    parse_stats = parse_blocks(json_path, pdf, item_id)
    log(f"      {parse_stats['blocks']} 个块，{parse_stats['figures']} 张图")

    # 标题四级兜底：显式填写 > CSV 已有 > 论文 doc_title 自动提取 > 文件名。
    # 提取必须在 parse 之后（标题藏在解析出的块里），且在切块之前定稿——
    # 父块回填、QC4 检索用的都是这个标题。
    title = title or csv_title or extract_title_from_blocks(item_id) or pdf.stem
    log(f"论文标题：{title[:80]}")

    log("[3/5] 父子切块…")
    meta = {
        "item_id": item_id,
        "title": title,
        "reader": reader,
        "doi": "",
    }
    parents, children = chunk_item(item_id, meta)
    log(f"      {len(children)} 个子块，{len(parents)} 个父块")

    # 在碰 Milvus 前先持久化旧父/子块。同步 Worker 和手动上传共用同一把
    # 跨进程锁；即便 API 与 Worker 同时收到任务，也只能串行替换语料。
    with PARENTS_FILE.open("r", encoding="utf-8") as file:
        old_parents = json.load(file)
    old_parents_backup = json.dumps(old_parents, ensure_ascii=False)
    old_lines = CHUNKS_FILE.read_text(encoding="utf-8").splitlines() if CHUNKS_FILE.exists() else []
    old_lines_backup = list(old_lines)
    old_children = [
        json.loads(line)
        for line in old_lines_backup
        if line.strip() and json.loads(line).get("item_id") == item_id
    ]

    log("[4/5] Milvus 更新…")
    deleted = delete_item_chunks(store, item_id)
    try:
        inserted = store.insert_chunks(children)
    except Exception as exc:
        delete_item_chunks(store, item_id)
        if old_children:
            store.insert_chunks(old_children)
        raise IngestionError(f"Milvus 写入失败，旧版本已恢复：{exc}") from exc
    log(f"      删旧 {deleted} 行，插入 {inserted} 行")

    new_parents = merge_parents_by_item(old_parents, item_id, parents)
    with PARENTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(new_parents, file, ensure_ascii=False)

    new_lines = merge_chunks_lines(old_lines, item_id, children)
    CHUNKS_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    log("[5/5] 质检…")
    failures = []

    if len(children) <= 0:
        failures.append("QC1 子块数为 0")

    got_back = item_chunk_count(store, item_id)
    if got_back != len(children):
        failures.append(f"QC2 Milvus 查回 {got_back} 行 != 插入 {len(children)} 行")

    consistent, detail = qc_parent_child_consistent(parents, children)
    if not consistent:
        failures.append(f"QC3 {detail}")

    # QC4 必须与生产检索同语义（默认 translate=True）：语料九成是英文，
    # 中文标题不翻译时 BM25 失效、纯向量被同主题中文论文压制，会误报
    # 「检索不命中」（实测 045 标题检索翻成英文后重排分 0.9947 满分命中）。
    smoke = store.retrieve(title, top_k=3, expand_parents=False)
    hit_ids = {chunk.metadata.get("item_id") for chunk in smoke.chunks}
    if item_id not in hit_ids:
        failures.append(f"QC4 标题检索 top3 未命中本篇（命中：{hit_ids or '无'}）")

    # QC5 存量不动：抽查 3 篇其他论文的行数前后一致（query 强一致）。
    for sid in sample_ids:
        after = item_chunk_count(store, sid)
        if after != sample_before[sid]:
            failures.append(f"QC5 存量被波及：{sid} 行数 {sample_before[sid]} -> {after}")

    if failures:
        for failure in failures:
            log(f"  ✗ {failure}")
        delete_item_chunks(store, item_id)
        if old_children:
            restored = store.insert_chunks(old_children)
            if restored != len(old_children):
                failures.append(
                    f"旧版本向量恢复不完整：{restored}/{len(old_children)}"
                )
        PARENTS_FILE.write_text(old_parents_backup, encoding="utf-8")
        CHUNKS_FILE.write_text("\n".join(old_lines_backup) + ("\n" if old_lines_backup else ""), encoding="utf-8")
        log("已回滚 Milvus 与本地合并文件，本次入库未生效。")
        raise IngestionError("质检失败，已回滚：" + "；".join(failures), failures=failures)
    log("      QC1-QC5 全部通过")

    # 登记：manifest + 索引 CSV + overrides
    manifest[item_id] = {
        "item_id": item_id,
        "pdf_sha256": pdf_hash,
        "pdf_file": pdf.name,
        "ingested_at": datetime.now().isoformat(timespec="seconds"),
        "chunk_count": len(children),
        "parent_count": len(parents),
        "collection": store.collection,
    }
    save_manifest(MANIFEST_FILE, manifest)

    # 可选字段留空时沿用库里已有值：强制重入库只补填的信息，
    # 不能把之前登记的整理者/主题覆盖成空。
    fields = {
        "title": title,
        "reader": reader or (existing_row or {}).get("reader", ""),
        "paper_file": pdf.name,
        "status": "ready",
        "theme": theme or (existing_row or {}).get("theme", ""),
    }
    csv_rows, action = upsert_csv_row(csv_rows, item_id, fields)
    save_index_rows(csv_rows)
    append_override(item_id, fields)

    return {
        "item_id": item_id,
        "status": "ingested",
        "inserted": inserted,
        "deleted": deleted,
        "chunk_count": len(children),
        "parent_count": len(parents),
        "sample_ids": sample_ids,
        "csv_action": action,
        "qc": "passed",
    }


def run_ingestion(
    pdf: Path,
    *,
    item_id: str | None = None,
    title: str = "",
    reader: str = "",
    theme: str = "",
    force: bool = False,
    log=print,
) -> dict:
    """带跨进程锁的兼容入口，CLI 与 FastAPI 原调用方式保持不变。"""

    from filelock import FileLock

    lock_path = PROJECT_DIR / "data" / "runtime" / "ingest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path), timeout=1800):
        return _run_ingestion_unlocked(
            pdf,
            item_id=item_id,
            title=title,
            reader=reader,
            theme=theme,
            force=force,
            log=log,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="单篇增量入库")
    parser.add_argument("pdf", help="论文 PDF 路径")
    parser.add_argument("--item-id", default=None, help="文献编号（缺省自动匹配/分配）")
    parser.add_argument("--title", default="", help="论文标题")
    parser.add_argument("--reader", default="", help="阅读整理者")
    parser.add_argument("--theme", default="", help="主题")
    parser.add_argument("--force", action="store_true", help="同哈希也强制重入库")
    args = parser.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"PDF 不存在：{pdf}")
        return 1

    try:
        report = run_ingestion(
            pdf,
            item_id=args.item_id,
            title=args.title,
            reader=args.reader,
            theme=args.theme,
            force=args.force,
        )
    except IngestionError as exc:
        print(f"入库失败：{exc}")
        return 1
    except Exception as exc:
        print(f"入库异常：{type(exc).__name__}：{exc}")
        return 1

    if report["status"] == "skipped":
        return 0

    print()
    print("=" * 60)
    print(f"入库完成：item_id={report['item_id']}")
    print(f"  子块 {report['inserted']}（删旧 {report['deleted']}），父块 {report['parent_count']}")
    print(f"  存量抽查 {report['sample_ids']} 行数前后一致")
    print(f"  索引 CSV：{report['csv_action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
