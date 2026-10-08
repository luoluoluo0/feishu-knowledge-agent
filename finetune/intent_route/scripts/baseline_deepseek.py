"""基线一：DeepSeek 现网分类跑冻结测试集。

直接复用 app/intent.py 的 classify_intent——包括它内部的降级逻辑
（任何失败降级为 simple_qa），这才是「现网口径」：微调模型要追平的
就是这条链路的数字，而不是理想化的纯 prompt。

产出：
- outputs/baseline_deepseek.jsonl   每行一条 (question, gold, pred, correct, latency_ms)
- outputs/baseline_deepseek_report.json  accuracy / macro-F1 / 每类 P/R/F1 / 混淆对 / 平均延迟

用法：
  python baseline_deepseek.py [--workers 6]
中断可续跑（按已处理行数跳过——测试集顺序固定，追加式写盘）。
"""

import argparse
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

INTENTS = ["simple_qa", "metadata_query", "summary_paper", "summary_ppt", "group_report", "compare_papers"]
TEST_PATH = Path(__file__).resolve().parents[1] / "data" / "splits" / "test.jsonl"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
RAW_PATH = OUT_DIR / "baseline_deepseek.jsonl"
REPORT_PATH = OUT_DIR / "baseline_deepseek_report.json"


def macro_f1(per_class: dict) -> float:
    import math

    f1s = [m["f1"] for m in per_class.values()
           if m["f1"] is not None and not math.isnan(m["f1"])]
    return sum(f1s) / len(f1s) if f1s else 0.0


def evaluate(rows: list[dict]) -> dict:
    tp = Counter()
    fp = Counter()
    fn = Counter()
    confusion = Counter()
    correct = sum(1 for r in rows if r["pred"] == r["gold"])
    for r in rows:
        gold, pred = r["gold"], r["pred"]
        if pred == gold:
            tp[gold] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1
            confusion[(gold, pred)] += 1

    per_class = {}
    for intent in INTENTS:
        precision = tp[intent] / (tp[intent] + fp[intent]) if tp[intent] + fp[intent] else float("nan")
        recall = tp[intent] / (tp[intent] + fn[intent]) if tp[intent] + fn[intent] else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision == precision and recall == recall and precision + recall
            else float("nan")
        )
        per_class[intent] = {
            "precision": round(precision, 4) if precision == precision else None,
            "recall": round(recall, 4) if recall == recall else None,
            "f1": round(f1, 4) if f1 == f1 else None,
            "support": tp[intent] + fn[intent],
        }

    latencies = sorted(r["latency_ms"] for r in rows)
    return {
        "n": len(rows),
        "accuracy": round(correct / len(rows), 4) if rows else None,
        "macro_f1": round(macro_f1(per_class), 4),
        "per_class": per_class,
        "top_confusions": [
            {"gold": g, "pred": p, "count": n}
            for (g, p), n in confusion.most_common(10)
        ],
        "latency_ms_avg": round(sum(latencies) / len(latencies)) if latencies else None,
        "latency_ms_p50": latencies[len(latencies) // 2] if latencies else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    # 延迟导入：classify_intent 只在真正跑基线时需要（依赖项目 app 包与 .env）。
    # 本文件同时被 AutoDL 上的评测脚本当指标库 import（evaluate/INTENTS 是纯函数，
    # 不能让顶层 import 把那边卡死）。
    from app.intent import classify_intent

    tests = [json.loads(l) for l in open(TEST_PATH, encoding="utf-8") if l.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 断点续跑：已有结果的按行数跳过（测试集顺序固定，追加式写盘）
    done_rows = []
    if RAW_PATH.exists():
        done_rows = [json.loads(l) for l in open(RAW_PATH, encoding="utf-8") if l.strip()]
    done = len(done_rows)
    todo = tests[done:]
    print(f"测试集 {len(tests)} 条，已完成 {done}，本次跑 {len(todo)}")

    lock = threading.Lock()
    raw_f = open(RAW_PATH, "a", encoding="utf-8")

    def run_one(item: dict) -> None:
        t0 = time.perf_counter()
        try:
            result = classify_intent(item["question"])
            pred = result.intent
        except Exception as exc:
            pred = f"__error__:{type(exc).__name__}"
        ms = int((time.perf_counter() - t0) * 1000)
        with lock:
            raw_f.write(
                json.dumps(
                    {"question": item["question"], "gold": item["intent"], "pred": pred, "latency_ms": ms},
                    ensure_ascii=False,
                )
                + "\n"
            )
            raw_f.flush()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, item) for item in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}，{time.time() - t0:.0f}s")
    raw_f.close()

    rows = [json.loads(l) for l in open(RAW_PATH, encoding="utf-8") if l.strip()]
    errors = [r for r in rows if str(r["pred"]).startswith("__error__")]
    if errors:
        print(f"警告：{len(errors)} 条调用报错，计入准确率前先看这些错误")
        report = evaluate([r for r in rows if not str(r["pred"]).startswith("__error__")])
        report["n_errors"] = len(errors)
    else:
        report = evaluate(rows)
        report["n_errors"] = 0

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\naccuracy={report['accuracy']}  macro_f1={report['macro_f1']}  "
          f"延迟 avg={report['latency_ms_avg']}ms p50={report['latency_ms_p50']}ms")
    print("主要混淆：")
    for c in report["top_confusions"][:5]:
        print(f"  {c['gold']} -> {c['pred']}: {c['count']}")
    print(f"报告: {REPORT_PATH}")


if __name__ == "__main__":
    main()
