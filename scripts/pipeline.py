"""全量重建编排：把手动脚本串成单一 CLI，失败即停。

用法：
    python scripts/pipeline.py --mode full

增量入库不走这里，用 ingest_paper.py（单篇、存量不动）。
不引 Airflow/Prefect 的理由：它们解决的是调度、多机协同与复杂依赖图，
当前是单机顺序流程，引入纯属运维负担。什么时候需要 Airflow——多数据源
定时汇入、失败重试策略、团队协作审批——到那时候再上。
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_DIR / "scripts"

# 全量重建的阶段序列：每个阶段是独立脚本，前一个失败后面不跑。
FULL_STAGES = [
    ("MinerU 批量解析（PDF -> middle_json）", [str(SCRIPTS / "run_mineru_batch.py")]),
    ("规范化块（middle_json -> jsonl）", [str(SCRIPTS / "parse_mineru.py")]),
    ("父子切块（jsonl -> chunks_v2）", [str(SCRIPTS / "build_chunks_v2.py")]),
    ("Milvus 全量重建（chunks_v2 -> 集合 + parents.json）", [str(SCRIPTS / "build_milvus_db.py")]),
]


def run_stage(name: str, command: list[str]) -> None:
    print()
    print("=" * 66)
    print(f"阶段：{name}")
    print("=" * 66)
    started = time.perf_counter()
    result = subprocess.run([sys.executable, *command])
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        print(f"\n✗ 阶段失败（{name}），耗时 {elapsed:.0f} 秒，退出码 {result.returncode}。")
        raise SystemExit(result.returncode)
    print(f"\n✓ 阶段完成：{name}（{elapsed:.0f} 秒）")


def final_sanity() -> None:
    """末尾抽查：集合行数 + 单篇检索命中，证明全量重建可用。"""

    print()
    print("=" * 66)
    print("最终抽查")
    print("=" * 66)
    sys.path.insert(0, str(PROJECT_DIR))
    from app.milvus_store import MilvusStore
    from app.config import get_settings

    store = MilvusStore(get_settings())
    print(f"集合行数：{store.count()}")
    chunks = store.search("多维健康贫困 研究方法", top_k=3)
    if not chunks:
        print("✗ 抽样检索无结果，全量重建可能有问题。")
        raise SystemExit(1)
    for chunk in chunks:
        print(f"  命中 [{chunk.metadata.get('item_id')}] {chunk.metadata.get('title', '')[:40]}")
    print("✓ 抽样检索正常。")


def main() -> int:
    parser = argparse.ArgumentParser(description="全量重建编排")
    parser.add_argument("--mode", choices=["full"], default="full")
    args = parser.parse_args()

    started = time.perf_counter()
    for name, command in FULL_STAGES:
        run_stage(name, command)
    final_sanity()
    print(f"\n全量重建完成，总耗时 {(time.perf_counter() - started) / 60:.1f} 分钟。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
