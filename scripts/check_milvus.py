import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from app.config import get_settings
from app.milvus_store import MilvusStore, ParentStore


# 检查 Milvus 链路：连接、数据量、检索、父子回填。
#
# 不写任何数据，只读。建库之后跑一遍确认结果。

DEFAULT_QUESTIONS = [
    "How were the sample counties and households selected?",
    "What is the theoretical framework of the study?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 Milvus 向量库链路")
    parser.add_argument("question", nargs="*", help="自定义检索问题")
    parser.add_argument("--top-k", type=int, default=4, help="返回条数，默认 4")
    args = parser.parse_args()

    settings = get_settings()
    questions = args.question or DEFAULT_QUESTIONS

    print(f"Milvus  ：{settings.milvus_uri}")
    print(f"集合    ：{settings.milvus_collection}")
    print("=" * 66)

    store = MilvusStore(settings)

    if not store.has_collection():
        print("集合不存在。请先运行 scripts/build_milvus_db.py")
        return 1

    print(f"服务端版本：{store.client.get_server_version()}")
    print(f"集合条数  ：{store.count()}")

    parents = ParentStore()
    print(f"父块条数  ：{len(parents)}  （来自 {parents.path.name}）")
    print()

    for question in questions:
        print(f"问：{question}")
        hits = store.search(question, top_k=args.top_k)

        if not hits:
            print("  （没有检索到内容）")
            print()
            continue

        for hit in hits:
            meta = hit.metadata
            section = str(meta.get("section", ""))[:34]
            print(
                f"  {hit.score:.3f}  [{meta.get('block_type', '?'):<8}]"
                f" p{meta.get('page', '?')}  {section}"
            )
            preview = str(hit.text).replace("\n", " ")[:88]
            print(f"         {preview}")

        # 展示父子回填：命中子块后能否取回完整上下文。
        parent_id = hits[0].metadata.get("parent_id", "")
        parent = parents.get(parent_id) if parent_id else None
        print()
        if parent:
            print(f"  父子回填 → {parent_id}「{parent.get('section', '')}」")
            print(
                f"    页 {parent.get('page')}-{parent.get('page_end')}"
                f"   含 {len(parent.get('child_ids', []))} 个子块"
                f"   共 {len(parent.get('text', ''))} 字"
            )
        else:
            print(f"  父子回填 → 找不到父块 {parent_id!r}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
