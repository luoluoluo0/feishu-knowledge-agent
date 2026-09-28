"""校准按意图分档的 rerank 门槛。

对评测集的各类问题直接跑 retrieve()（重排开着、不做拒答判断），
记录每类问题的 top rerank 分数分布，回答两个问题：

1. 总结类（summary_paper / group_report / compare_papers）语料内的
   分数低位在哪——总结档门槛必须低于它，否则照样误拒；
2. refusal 类（语料外）有多少会被意图识别分进总结档——那是分档
   方案唯一的风险敞口：这些题拿到低门槛后，分数若还高于总结档
   门槛，拒答头就不会挂。实测分布出来才知道要不要加保险。

用法：python scripts/calibrate_rerank_thresholds.py [--eval eval_set.jsonl]
结果同时打印到 stdout，供 dev-notes 归档。
"""

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from app.config import get_settings  # noqa: E402
from app.intent import classify_intent  # noqa: E402
from app.milvus_store import MilvusStore  # noqa: E402
from app.query_context import SUMMARY_INTENTS  # noqa: E402


logger = logging.getLogger(__name__)


# 各评测类别对应的意图，与 run_eval.py 的 CATEGORY_INTENT 保持一致。
# refusal 特殊：它是「结果」不是「意图」，实际会被分成 simple_qa 或
# metadata_query——校准要实测有没有被分进总结档的。
CATEGORY_INTENT = {
    "summary_paper": "summary_paper",
    "compare_papers": "compare_papers",
    "group_report": "group_report",
    "en_fact": "simple_qa",
    "zh_fact": "simple_qa",
    "refusal": "(实测)",
}

CATEGORIES = list(CATEGORY_INTENT)


def percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(len(sorted_values) * ratio))
    return sorted_values[index]


def load_cases(eval_path: Path) -> dict[str, list[dict]]:
    by_category: dict[str, list[dict]] = {}
    with eval_path.open("r", encoding="utf-8") as file:
        for line in file:
            case = json.loads(line)
            if case.get("turns"):
                # 多轮题的 query 不能独立使用，检索校准跳过。
                continue
            if case.get("category") in CATEGORIES:
                by_category.setdefault(case["category"], []).append(case)
    return by_category


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="eval_set.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    eval_path = Path(args.eval)
    if not eval_path.is_absolute() and not eval_path.exists():
        eval_path = PROJECT_DIR / "data" / "eval" / args.eval

    settings = get_settings()
    store = MilvusStore(settings)

    by_category = load_cases(eval_path)

    # scores[category] = [top rerank 分数]
    scores: dict[str, list[float]] = {}
    refusal_intents: dict[str, int] = {}

    for category in CATEGORIES:
        cases = by_category.get(category, [])
        values: list[float] = []
        for index, case in enumerate(cases, start=1):
            query = case.get("query", "")
            if not query:
                continue

            try:
                result = store.retrieve(query, top_k=args.top_k)
            except Exception as exc:
                logger.warning("第 %d 条检索失败，跳过：%s", index, exc)
                continue
            if result.reranked and result.top_score is not None:
                values.append(result.top_score)

            if category == "refusal":
                intent = classify_intent(query, settings=settings).intent
                refusal_intents[intent] = refusal_intents.get(intent, 0) + 1

            if index % 10 == 0:
                print(f"  {category}: {index}/{len(cases)}", flush=True)

        scores[category] = values

    print("\n" + "=" * 72)
    print(f"{'类别':<16}{'条数':>6}{'min':>9}{'p10':>9}{'中位':>9}{'max':>9}")
    print("-" * 72)
    for category in CATEGORIES:
        values = sorted(scores.get(category, []))
        if not values:
            print(f"{category:<16}{0:>6}")
            continue
        print(
            f"{category:<16}{len(values):>6}"
            f"{values[0]:>9.4f}{percentile(values, 0.10):>9.4f}"
            f"{statistics.median(values):>9.4f}{values[-1]:>9.4f}"
        )

    summary_all: list[float] = []
    for category in ("summary_paper", "group_report", "compare_papers"):
        summary_all.extend(scores.get(category, []))
    summary_all.sort()

    print("\n总结档分数低位（三个类别合并，共 %d 条）：" % len(summary_all))
    for ratio in (0.0, 0.05, 0.10, 0.25):
        print(f"  p{int(ratio * 100):02d} = {percentile(summary_all, ratio):.4f}")

    print("\n候选门槛下的通过率（语料内总结题 / 语料外 refusal 题）：")
    refusal_values = sorted(scores.get("refusal", []))
    for candidate in (0.10, 0.20, 0.30, 0.35, 0.40, 0.50):
        pass_summary = sum(1 for v in summary_all if v >= candidate)
        pass_refusal = sum(1 for v in refusal_values if v >= candidate)
        print(
            f"  门槛 {candidate:.2f}: 总结题通过 "
            f"{pass_summary}/{len(summary_all)}"
            f"（{pass_summary / max(1, len(summary_all)):.1%}），"
            f"refusal 通过 {pass_refusal}/{len(refusal_values)}"
            f"（{pass_refusal / max(1, len(refusal_values)):.1%}）"
        )

    print("\nrefusal 类问题的实际意图分布（分档风险敞口）：")
    for intent, count in sorted(refusal_intents.items(), key=lambda kv: -kv[1]):
        risk = " ← 会拿到总结档低门槛" if intent in SUMMARY_INTENTS else ""
        print(f"  {intent}: {count}{risk}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
