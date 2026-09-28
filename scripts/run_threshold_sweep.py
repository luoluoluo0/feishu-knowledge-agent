"""跑一遍 rerank 阈值扫描，把曲线数据落盘给 /admin/observe 画图。

对评测集的单轮题（可答 5 类 + 语料外 refusal 类）各跑 retrieve()，
拿每题的 top rerank 分数，然后扫描阈值 t ∈ [0, 1]：

- 通过率(t)   ：可答题分数 ≥ t 的占比——门槛不该把能答的挡在外面；
- 拒答正确率(t)：refusal 题分数 < t 的占比——语料外的就该被拦住；
- Youden(t) = 通过率 + 拒答正确率 − 1，取最大处为建议工作点。

产出 data/eval/threshold_sweep.json：曲线点 + 建议工作点 + 各类分数
分布。/admin/observe 读这份文件画图——检索 170 题要几分钟，不能在
HTTP 请求里现算，重跑本脚本即可刷新。

用法：python scripts/run_threshold_sweep.py
"""

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from app.config import get_settings  # noqa: E402
from app.milvus_store import MilvusStore  # noqa: E402

EVAL_DIR = PROJECT_DIR / "data" / "eval"
SWEEP_PATH = EVAL_DIR / "threshold_sweep.json"

# 参与扫描的类别：可答 5 类（有金标准文献可检索）+ 语料外 refusal。
ANSWERABLE_CATEGORIES = ["summary_paper", "compare_papers", "group_report", "en_fact", "zh_fact"]
REFUSAL_CATEGORY = "refusal"

TOP_K = 5


def load_cases() -> dict[str, list[dict]]:
    by_category: dict[str, list[dict]] = {}
    with (EVAL_DIR / "eval_set.jsonl").open("r", encoding="utf-8") as file:
        for line in file:
            case = json.loads(line)
            # 多轮题的 query 不能独立使用，扫描跳过（与标定脚本一致）。
            if case.get("turns"):
                continue
            if case.get("category") in ANSWERABLE_CATEGORIES + [REFUSAL_CATEGORY]:
                by_category.setdefault(case["category"], []).append(case)
    return by_category


def collect_scores(store: MilvusStore, cases: list[dict], label: str) -> list[float]:
    values: list[float] = []
    for index, case in enumerate(cases, start=1):
        query = case.get("query", "")
        if not query:
            continue
        try:
            result = store.retrieve(query, top_k=TOP_K)
        except Exception as exc:
            print(f"  [{label}] 第 {index} 条检索失败，跳过：{exc}", flush=True)
            continue
        if result.reranked and result.top_score is not None:
            values.append(float(result.top_score))
        if index % 10 == 0:
            print(f"  {label}: {index}/{len(cases)}", flush=True)
    return values


def main() -> int:
    settings = get_settings()
    store = MilvusStore(settings)
    by_category = load_cases()

    print(f"扫描对象：可答 {len(ANSWERABLE_CATEGORIES)} 类 + refusal，共 "
          f"{sum(len(c) for c in by_category.values())} 条单轮题", flush=True)

    scores: dict[str, list[float]] = {}
    for category in ANSWERABLE_CATEGORIES + [REFUSAL_CATEGORY]:
        scores[category] = collect_scores(store, by_category.get(category, []), category)

    answerable: list[float] = []
    for category in ANSWERABLE_CATEGORIES:
        answerable.extend(scores[category])
    refusal = scores[REFUSAL_CATEGORY]

    curve = []
    best = {"t": 0.0, "pass_rate": 0.0, "reject_rate": 0.0, "youden": -1.0}
    for step in range(0, 101):
        t = step / 100
        pass_rate = sum(1 for v in answerable if v >= t) / len(answerable) if answerable else 0.0
        reject_rate = sum(1 for v in refusal if v < t) / len(refusal) if refusal else 0.0
        youden = round(pass_rate + reject_rate - 1, 4)
        curve.append({"t": round(t, 2), "pass_rate": round(pass_rate, 4),
                      "reject_rate": round(reject_rate, 4), "youden": youden})
        if youden > best["youden"]:
            best = {"t": round(t, 2), "pass_rate": round(pass_rate, 4),
                    "reject_rate": round(reject_rate, 4), "youden": youden}

    def dist(values: list[float]) -> dict:
        v = sorted(values)
        if not v:
            return {"n": 0}
        n = len(v)

        def pct(ratio: float) -> float:
            return round(v[min(n - 1, int(n * ratio))], 4)

        return {"n": n, "min": round(v[0], 4), "p25": pct(0.25),
                "median": round(statistics.median(v), 4),
                "p75": pct(0.75), "max": round(v[-1], 4)}

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rerank_model": settings.rerank_model,
        "top_k": TOP_K,
        "answerable": dist(answerable),
        "refusal": dist(refusal),
        "by_category": {c: dist(scores[c]) for c in ANSWERABLE_CATEGORIES + [REFUSAL_CATEGORY]},
        "curve": curve,
        "chosen": best,
    }
    SWEEP_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n可答 {len(answerable)} 条 | refusal {len(refusal)} 条")
    print(f"建议工作点 t={best['t']:.2f}：通过率 {best['pass_rate']:.1%}，"
          f"拒答正确率 {best['reject_rate']:.1%}，Youden {best['youden']:.3f}")
    print(f"已写入 {SWEEP_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
