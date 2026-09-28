"""zh_fact 中文检索退化的诊断实验。

四种检索变体，回答「BM25 分支该用什么查询」：

  A. 现状      ：dense=中文原文，sparse=英文译文（线上行为）
  B. 纯向量     ：只有 dense=中文原文（BM25 完全不参与）
  C. 拼接 BM25  ：sparse=中文原文 + 空格 + 英文译文（一次请求，两种语言的
                  词都在查询里；icu 分词器各自匹配各自语言的文档）
  D. 三路       ：dense=中文 + sparse=英文译文 + sparse=中文原文（三路独立
                  RRF；Milvus 同字段双请求是否允许待验证）

判分：expected_item_ids 被 top-5 召回的比例（与 run_eval retrieval 模式同口径）。
对照集：zh_fact（要修的）、cross_lingual（不能改坏的）、en_fact（不受影响的）。
"""

import json
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from pymilvus import AnnSearchRequest, RRFRanker

from app.config import get_settings
from app.milvus_store import OUTPUT_FIELDS, RRF_K, MilvusStore, SPARSE_FIELD, to_query_text
from app.query_translation import translate_to_english

EVAL_PATH = PROJECT_DIR / "data" / "eval" / "eval_set.jsonl"
CATEGORIES = ("zh_fact", "cross_lingual", "en_fact")
TOP_K = 5
RECALL_LIMIT = 20


def load_cases() -> dict[str, list[dict]]:
    by_category: dict[str, list[dict]] = defaultdict(list)
    with EVAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            case = json.loads(line)
            if case.get("turns"):
                continue
            if case.get("category") in CATEGORIES and case.get("expected_item_ids"):
                by_category[case["category"]].append(case)
    return by_category


def recall_of(item_ids: list[str], expected: list[str]) -> float:
    if not expected:
        return 0.0
    return len(set(item_ids) & set(expected)) / len(set(expected))


def make_variants(store: MilvusStore, question: str, english: str | None):
    """返回 {变体名: 召回函数}。查询向量只算一次，各变体复用。"""

    vector = store.embeddings.embed_query(to_query_text(question))

    def dense_req(expr):
        return AnnSearchRequest(
            data=[vector],
            anns_field="vector",
            param={"metric_type": "COSINE"},
            limit=RECALL_LIMIT,
            expr=expr,
        )

    def sparse_req(text: str, expr):
        return AnnSearchRequest(
            data=[text],
            anns_field=SPARSE_FIELD,
            param={"metric_type": "BM25"},
            limit=RECALL_LIMIT,
            expr=expr,
        )

    def run(reqs) -> list[str]:
        results = store.client.hybrid_search(
            collection_name=store.collection,
            reqs=reqs,
            ranker=RRFRanker(k=RRF_K),
            limit=TOP_K,
            output_fields=OUTPUT_FIELDS,
        )
        out = []
        for hit in results[0] if results else []:
            entity = hit.get("entity", {}) or {}
            out.append(str(entity.get("item_id", "")))
        return out

    def variant_a() -> list[str]:
        """现状：dense 中文 + sparse 英文。"""
        return run([dense_req(None), sparse_req(english, None)])

    def variant_b() -> list[str]:
        """纯向量。"""
        results = store.client.search(
            collection_name=store.collection,
            data=[vector],
            anns_field="vector",
            limit=TOP_K,
            output_fields=OUTPUT_FIELDS,
        )
        out = []
        for hit in results[0] if results else []:
            entity = hit.get("entity", {}) or {}
            out.append(str(entity.get("item_id", "")))
        return out

    def variant_c() -> list[str]:
        """拼接：sparse = 中文原文 + 空格 + 英文译文。"""
        concat = f"{question} {english}" if english else question
        return run([dense_req(None), sparse_req(concat, None)])

    def variant_d() -> list[str]:
        """三路：dense 中文 + sparse 英文 + sparse 中文。"""
        return run([dense_req(None), sparse_req(english, None), sparse_req(question, None)])

    return {"A现状": variant_a, "B纯向量": variant_b, "C拼接": variant_c, "D三路": variant_d}


def main() -> int:
    settings = get_settings()
    store = MilvusStore(settings)
    if not store.has_collection():
        print("集合不存在")
        return 1

    by_category = load_cases()
    jobs = [(cat, case) for cat, cases in by_category.items() for case in cases]
    total = len(jobs)
    print(f"待诊断 {total} 条：", {k: len(v) for k, v in by_category.items()})

    stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    d_error: list[str] = []
    done = 0
    lock = threading.Lock()

    def _work(category: str, case: dict) -> None:
        nonlocal done
        query = case["query"]
        expected = sorted(set(case["expected_item_ids"]))
        english = translate_to_english(query, settings=settings).english
        variants = make_variants(store, query, english)
        local: dict[str, list[float]] = defaultdict(list)
        for name, fn in variants.items():
            try:
                local[name].append(recall_of(fn(), expected))
            except Exception as exc:
                local[name].append(-1.0)
                if name == "D三路" and not d_error:
                    d_error.append(f"{type(exc).__name__}: {exc}"[:140])
        with lock:
            for name, vals in local.items():
                stats[category][name].extend(vals)
            done += 1
            if done % 15 == 0:
                print(f"  进度 {done}/{total}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for cat, case in jobs:
            pool.submit(_work, cat, case)

    print("\n" + "=" * 72)
    print(f"{'类别':<16}{'A现状':>10}{'B纯向量':>10}{'C拼接':>10}{'D三路':>10}")
    print("-" * 72)
    for cat in CATEGORIES:
        row = []
        for name in ("A现状", "B纯向量", "C拼接", "D三路"):
            vals = [v for v in stats[cat].get(name, []) if v >= 0]
            row.append(f"{sum(vals)/len(vals):.3f}" if vals else "N/A")
        print(f"{cat:<16}" + "".join(f"{v:>10}" for v in row))
    if d_error:
        print("D三路错误:", d_error[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
