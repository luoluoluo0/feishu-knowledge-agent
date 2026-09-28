import argparse
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

try:
    from app.config import get_settings
    from app.milvus_store import SCALAR_FIELDS, MilvusStore
except ModuleNotFoundError as exc:
    # 项目依赖装在 base 环境，MinerU 的 GPU 依赖装在 torch5060。
    # 用错环境时报错信息只有一句「No module named xxx」，很难定位。
    print(f"缺少依赖：{exc.name}")
    print()
    print("项目脚本要用装了依赖的那个 Python 跑，通常是 base：")
    print("  conda activate base")
    print(f"  python {Path(__file__).name}")
    print()
    print("（MinerU 相关脚本才需要用 torch5060）")
    raise SystemExit(1)


# 把现有集合重建成带 BM25 全文检索的新集合。
#
# 为什么要重建：BM25 需要三样东西配套——开了分析器的 text 字段、
# 稀疏向量字段、以及把两者绑起来的 BM25 函数。前两样能给已有集合补，
# 但 BM25 函数只能在建集合时声明（实测 add_collection_function 会报
# "currently does not support adding BM25 function"）。所以只能重建。
#
# 好在不用重新解析 PDF，也不用重新切分，甚至不用重新算向量——
# Milvus 允许把向量字段读出来，直接搬过去就行。
#
# 新集合用另一个名字，旧的保留。验证通过后改 .env 里的
# MILVUS_COLLECTION 切换过去，旧的确认没用了再删。


def main() -> int:
    parser = argparse.ArgumentParser(description="重建集合以支持 BM25 混合检索")
    parser.add_argument("--source", default="", help="源集合名，默认取配置里的")
    parser.add_argument("--target", default="", help="目标集合名，默认 <源集合>_bm25")
    parser.add_argument(
        "--batch", type=int, default=500, help="写入批大小，默认 500"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="目标集合已存在时先删掉重建。上次跑到一半失败会留下半成品。",
    )
    args = parser.parse_args()

    settings = get_settings()
    source_name = args.source or settings.milvus_collection
    target_name = args.target or f"{source_name}_bm25"

    print(f"Milvus    ：{settings.milvus_uri}")
    print(f"源集合    ：{source_name}")
    print(f"目标集合  ：{target_name}")
    print("=" * 66)

    source = MilvusStore(settings, collection=source_name)
    if not source.has_collection():
        print(f"源集合不存在：{source_name}")
        return 1

    target = MilvusStore(settings, collection=target_name)
    if target.has_collection():
        if not args.force:
            print(f"目标集合已存在：{target_name}")
            print("要覆盖的话加 --force，或者先手工删掉：")
            print(f'  python -c "from pymilvus import MilvusClient; '
                  f"MilvusClient(uri='{settings.milvus_uri}').drop_collection('{target_name}')\"")
            return 1
        print(f"目标集合已存在，--force 已指定，先删掉重建。")
        target.drop()

    # ① 读源数据（含向量）
    print()
    print("① 读取源集合数据（含向量）……")
    started = time.perf_counter()
    rows = source.query_all()
    print(f"   读到 {len(rows)} 条，用时 {time.perf_counter() - started:.1f} 秒")

    if not rows:
        print("   源集合是空的，没什么可搬。")
        return 1

    has_vector = sum(1 for row in rows if row.get("vector"))
    print(f"   其中带向量的 {has_vector} 条")
    if has_vector < len(rows):
        print(f"   注意：有 {len(rows) - has_vector} 条没有向量，搬过去也检索不到。")

    # ② 建新集合（带 BM25）
    print()
    print("② 建新集合（开启分析器 + 稀疏字段 + BM25 函数）……")
    target.ensure_collection(recreate=False)
    print("   已建。")

    # ③ 写入
    # 注意不要带 sparse 字段：那是 BM25 函数根据 text 自动生成的，
    # 手工提供会冲突。源数据里本来也没有这个字段。
    print()
    print("③ 写入新集合……")
    # chunk_id 是主键，单独定义的，不在 SCALAR_FIELDS 里，必须显式带上。
    fields = ["chunk_id"] + [name for name, _ in SCALAR_FIELDS] + ["vector"]
    written = 0
    started = time.perf_counter()

    for start in range(0, len(rows), args.batch):
        batch = [
            {key: row.get(key) for key in fields}
            for row in rows[start : start + args.batch]
        ]
        target.client.insert(collection_name=target_name, data=batch)
        written += len(batch)
        print(f"   {written}/{len(rows)}", end="\r", flush=True)

    target.client.flush(target_name)
    elapsed = time.perf_counter() - started
    print(f"   写入 {written} 条，用时 {elapsed:.1f} 秒。")

    # ④ 验收
    print()
    print("=" * 66)
    print("④ 验收")
    print("=" * 66)

    target_rows = target.query_all(output_fields=["chunk_id"])
    print(f"   源集合 {len(rows)} 条   新集合 {len(target_rows)} 条")
    if len(target_rows) != len(rows):
        print("   ⚠ 条数对不上，检查一下。")

    print()
    print("   同一组问题，三种检索方式对比：")
    for question in ["生计韧性怎么衡量", "How are sample counties selected?"]:
        print()
        print(f"   问：{question}")
        for mode, label in [("dense", "纯向量"), ("rrf", "混合 RRF")]:
            if mode == "dense":
                hits = target.search(question, top_k=2)
            else:
                hits = target.hybrid_search(question, top_k=2)
            for hit in hits[:2]:
                text = str(hit.text).replace("\n", " ")[:56]
                print(f"     {label:<7} {hit.score:>8.4f}  {text}")

    print()
    print("=" * 66)
    print("下一步")
    print("=" * 66)
    print(f"   1. 验证新集合检索正常：")
    print(f"      python scripts/run_eval.py --mode retrieval --collection {target_name}")
    print(f"   2. 确认没问题后，改 .env 切换：")
    print(f"      MILVUS_COLLECTION={target_name}")
    print(f"   3. 旧的确认不再需要了再删：")
    print(f'      python -c "from pymilvus import MilvusClient; '
          f"MilvusClient(uri='{settings.milvus_uri}').drop_collection('{source_name}')\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
