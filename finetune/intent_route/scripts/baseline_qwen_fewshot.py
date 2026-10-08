"""基线二：Qwen3-1.7B 基座 few-shot 跑冻结测试集（零训练对照）。

与 baseline_deepseek.py 完全同构：同一份 test.jsonl、同一套指标，
唯一差别是推理端点走 AutoDL 隧道（http://127.0.0.1:18000）的
vLLM OpenAI 兼容接口，模型为未训练的 Qwen3-1.7B 基座。

few-shot 示例只从 train 集取（每类 1 条，挑短的），绝不碰 test/val——
否则基线本身就被污染了。

用法：
  python baseline_qwen_fewshot.py [--endpoint http://127.0.0.1:18000] [--workers 6]
"""

import argparse
import json
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

INTENTS = ["simple_qa", "metadata_query", "summary_paper", "summary_ppt", "group_report", "compare_papers"]
TEST_PATH = DATA_DIR / "splits" / "test.jsonl"
TRAIN_PATH = DATA_DIR / "splits" / "train.jsonl"
RAW_PATH = OUT_DIR / "baseline_qwen_fewshot.jsonl"
REPORT_PATH = OUT_DIR / "baseline_qwen_fewshot_report.json"

SYSTEM = """你是论文资料问答系统的意图分类模块。判断用户问题属于哪一类意图，只输出 JSON。

可选意图：
1. simple_qa：普通问答，问某个事实、概念或结论。
2. metadata_query：元数据查询，问编号、整理者、DOI、附件、缺失情况、清单统计。
3. summary_paper：总结某篇论文，要研究问题、方法、数据、结论。
4. summary_ppt：总结某篇 PPT 或汇报内容。
5. group_report：生成组会汇报提纲。
6. compare_papers：对比多篇文献的异同。

输出格式：{"intent": "六选一", "reason": "简短理由"}"""

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def pick_few_shot(train_rows: list[dict]) -> list[dict]:
    """每类挑 1 条最短的真实种子做示例（短例子的示范焦点更集中）。"""
    messages = []
    by_intent = {}
    for r in train_rows:
        if r.get("source") == "distill":
            continue
        by_intent.setdefault(r["intent"], []).append(r)
    for intent in INTENTS:
        rows = by_intent.get(intent) or []
        if not rows:
            continue
        ex = min(rows, key=lambda r: len(r["question"]))
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": json.dumps({"intent": intent, "reason": "示例"}, ensure_ascii=False)})
    return messages


def parse_intent(text: str) -> str:
    text = THINK_RE.sub("", text)
    m = re.search(r'"intent"\s*:\s*"([a-z_]+)"', text)
    if m:
        return m.group(1)
    # 兜底：全文里找第一个出现的合法意图词
    for intent in INTENTS:
        if intent in text:
            return intent
    return "__unparsed__"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="qwen3-base")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    tests = [json.loads(l) for l in open(TEST_PATH, encoding="utf-8") if l.strip()]
    train_rows = [json.loads(l) for l in open(TRAIN_PATH, encoding="utf-8") if l.strip()]
    few_shot = pick_few_shot(train_rows)
    print(f"few-shot 示例 {len(few_shot) // 2} 条（均来自 train）")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done_rows = []
    if RAW_PATH.exists():
        done_rows = [json.loads(l) for l in open(RAW_PATH, encoding="utf-8") if l.strip()]
    todo = tests[len(done_rows):]
    print(f"测试集 {len(tests)} 条，已完成 {len(done_rows)}，本次跑 {len(todo)}")

    lock = threading.Lock()
    raw_f = open(RAW_PATH, "a", encoding="utf-8")

    def run_one(item: dict) -> None:
        t0 = time.perf_counter()
        pred = "__error__"
        try:
            resp = requests.post(
                f"{args.endpoint}/v1/chat/completions",
                json={
                    "model": args.model,
                    "messages": [{"role": "system", "content": SYSTEM}, *few_shot,
                                 {"role": "user", "content": item["question"]}],
                    "temperature": 0,
                    "max_tokens": 300,
                },
                timeout=120,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            pred = parse_intent(text)
        except Exception as exc:
            pred = f"__error__:{type(exc).__name__}"
        ms = int((time.perf_counter() - t0) * 1000)
        with lock:
            raw_f.write(json.dumps(
                {"question": item["question"], "gold": item["intent"], "pred": pred, "latency_ms": ms},
                ensure_ascii=False) + "\n")
            raw_f.flush()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, item) for item in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}，{time.time() - t0:.0f}s")
    raw_f.close()

    # 指标计算与 DeepSeek 基线同构
    sys.path.insert(0, str(Path(__file__).parent))
    from baseline_deepseek import evaluate

    rows = [json.loads(l) for l in open(RAW_PATH, encoding="utf-8") if l.strip()]
    errors = [r for r in rows if str(r["pred"]).startswith("__error__") or r["pred"] == "__unparsed__"]
    clean = [r for r in rows if r not in errors]
    report = evaluate(clean)
    report["n_errors_or_unparsed"] = len(errors)
    if errors:
        err_kinds = Counter(r["pred"] for r in errors)
        report["error_kinds"] = dict(err_kinds)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\naccuracy={report['accuracy']}  macro_f1={report['macro_f1']}  "
          f"延迟 avg={report['latency_ms_avg']}ms  失败/未解析={len(errors)}")
    print("主要混淆：")
    for c in report["top_confusions"][:5]:
        print(f"  {c['gold']} -> {c['pred']}: {c['count']}")
    print(f"报告: {REPORT_PATH}")


if __name__ == "__main__":
    main()
