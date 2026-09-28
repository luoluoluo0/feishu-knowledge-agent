"""把 VLM caption 拼进 Milvus 里的 figure 块并重新嵌入（方案 A 环节 3）。

figure 块已在库（222 块，text=原图注），本脚本做的事：
  1. 读 figure_captions.jsonl
  2. 从 Milvus 取出全部 block_type=figure 的块
  3. text 拼接为「原图注 + 图表解读（类型/轴/要点）」
  4. 重新嵌入，按 chunk_id upsert 回原集合

原图注随时可从 data/processed/chunks_v2.jsonl 恢复，upsert 幂等可重跑。

用法：
  python scripts/update_figure_chunks.py --dry-run   # 先看拼接结果不写库
  python scripts/update_figure_chunks.py             # 真正写库
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from pymilvus import MilvusClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.milvus_store import MilvusStore, to_embed_text  # noqa: E402

CAPTIONS_PATH = PROJECT_DIR / "data" / "processed" / "figure_captions.jsonl"

logger = logging.getLogger(__name__)


def format_enriched_text(orig_text: str, vlm: dict) -> str:
    """拼接增强后的块文本。

    布局考虑：原图注在最前（保留原始出处语义，中英都有）；
    解读部分用中文标签（用户查询是中文，BM25 与向量都吃中文词）；
    key_claims 用序号连接，保持句子边界清晰利于重排。
    """

    lines = [orig_text.strip(), "【图表解读】"]
    summary = str(vlm.get("summary") or "").strip()
    if summary:
        lines.append(summary)
    axes = vlm.get("axes") or {}
    meta = [f"类型：{vlm.get('chart_type', 'other')}"]
    if isinstance(axes, dict):
        if axes.get("x"):
            meta.append(f"横轴：{axes['x']}")
        if axes.get("y"):
            meta.append(f"纵轴：{axes['y']}")
    lines.append("；".join(meta))
    claims = [str(c).strip() for c in vlm.get("key_claims") or [] if str(c).strip()]
    if claims:
        lines.append("要点：" + "；".join(f"{i}）{c}" for i, c in enumerate(claims, 1)))
    return "\n".join(line for line in lines if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只打印拼接结果，不写库")
    args = parser.parse_args()

    if not CAPTIONS_PATH.exists():
        print(f"找不到 {CAPTIONS_PATH.name}，先跑 generate_figure_captions.py")
        return 1

    captions: dict[str, dict] = {}
    for line in CAPTIONS_PATH.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("vlm") and not row.get("error"):
            captions[str(row["image_path"])] = row

    settings = get_settings()
    store = MilvusStore(settings)
    if not store.has_collection():
        print("Milvus 集合不存在")
        return 1
    client = MilvusClient(uri=settings.milvus_uri)

    rows = client.query(
        collection_name=store.collection,
        filter='block_type == "figure"',
        output_fields=["*"],
        limit=1000,
    )
    print(f"库内 figure 块 {len(rows)} 个，可用 caption {len(captions)} 份")

    updated = skipped = failed = 0
    for row in rows:
        image_path = str(row.get("image_path") or "")
        cap = captions.get(image_path)
        if cap is None:
            skipped += 1
            continue

        new_text = format_enriched_text(str(row.get("text") or ""), cap["vlm"])
        if args.dry_run:
            print(f"--- {row['chunk_id']} ({image_path})")
            print(new_text)
            print()
            continue

        try:
            chunk = dict(row)  # 字段名与 build_row 的输入一一对应
            chunk["text"] = new_text
            vector = store.embeddings.embed_documents([to_embed_text(chunk)])[0]
            client.upsert(collection_name=store.collection, data=[store.build_row(chunk, vector)])
            updated += 1
        except Exception as exc:
            failed += 1
            logger.warning("upsert 失败 %s：%s", row.get("chunk_id"), exc)

    print(f"\n更新 {updated}，跳过 {skipped}（无 caption），失败 {failed}")
    if args.dry_run:
        print("（dry-run，未写库。确认无误后去掉 --dry-run 重跑）")
    else:
        print("验证：python scripts/check_milvus.py '哪篇论文有收入增长趋势的图'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
