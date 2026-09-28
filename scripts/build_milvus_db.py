import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from app.config import get_settings
from app.milvus_store import PARENT_STORE_FILE, MilvusStore


# 把切分结果（chunks_v2.jsonl）写进 Milvus。
#
# 子块进 Milvus（带向量），父块写成本地 JSON——
# 父块不参与向量检索，只在命中后按 id 查回，没必要占用向量库。
#
# 依赖：scripts/build_chunks_v2.py 的产出。
# 前置：Milvus 服务已启动（docker compose -f docker-compose.milvus.yml up -d）。

CHUNKS_FILE = PROJECT_DIR / "data" / "processed" / "chunks_v2.jsonl"


def load_chunks() -> tuple[list[dict], list[dict]]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"没有找到 {CHUNKS_FILE}\n请先运行 scripts/build_chunks_v2.py"
        )

    parents: list[dict] = []
    children: list[dict] = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("chunk_type") == "parent":
                parents.append(chunk)
            else:
                children.append(chunk)

    return parents, children


def save_parents(parents: list[dict]) -> None:
    PARENT_STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PARENT_STORE_FILE.open("w", encoding="utf-8") as file:
        json.dump(parents, file, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="把切分结果写进 Milvus")
    parser.add_argument(
        "--append",
        action="store_true",
        help="不清空集合，直接追加。默认会先删除重建，保证结果和 chunks_v2.jsonl 一致。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只写入前 N 条子块，用于快速验证。默认全部。",
    )
    args = parser.parse_args()

    settings = get_settings()
    parents, children = load_chunks()

    if args.limit > 0:
        children = children[: args.limit]

    print(f"切分文件：{CHUNKS_FILE}")
    print(f"Milvus  ：{settings.milvus_uri} / {settings.milvus_collection}")
    print(f"父块 {len(parents)}   子块 {len(children)}")
    print("=" * 60)

    if not children:
        print("没有可写入的子块。")
        return 1

    save_parents(parents)
    print(f"父块已写入：{PARENT_STORE_FILE}")

    store = MilvusStore(settings)
    store.ensure_collection(recreate=not args.append)

    before = store.count()
    if args.append:
        print(f"追加模式，集合现有 {before} 条")

    print(f"开始写入（批量接口），共 {len(children)} 条子块……")
    written = store.insert_chunks(children)
    print(f"本轮写入 {written} 条")

    if store.shrunk:
        print(f"缩短后才通过向量化 {len(store.shrunk)} 条（正常情况应为 0）")
        for chunk_id, ratio in store.shrunk[:5]:
            print(f"    {chunk_id}  缩短到 {int(ratio * 100)}%")

    if store.skipped:
        print(f"向量化失败被跳过 {len(store.skipped)} 条：{store.skipped[:10]}")

    after = store.count()
    print()
    print("=" * 60)
    print(f"集合内总条数：{after}")

    # 抽查一条，确认能按 item_id 过滤检索。
    sample_item = children[0].get("item_id", "")
    hits = store.search("研究方法", top_k=3, item_id=sample_item)
    print(f"抽查检索（item_id={sample_item}，top 3）：")
    for hit in hits:
        preview = hit.text[:60].replace("\n", " ")
        print(f"  {hit.score:.3f}  [{hit.metadata.get('block_type')}] {preview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
