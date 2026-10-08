"""端点版评测：零样本打 OpenAI 兼容端点（vLLM / Ollama / llama-server 通吃）。

用途：验证部署后的服务精度与训练口径一致——vLLM 起服务后跑一次，
accuracy 应与裸 transformers 的 99.67% 一致（±0.5pp 内算正常浮动，
偏差大说明服务配置或预处理有问题）。

与 eval_finetuned.py 的差别：不加载本地权重，只打 HTTP 端点——
所以能评任何部署形态（vLLM FP16 / Ollama Q4 / llama-server Q8），
同一把冻结尺子横向对比。

用法：
  python eval_endpoint.py --endpoint http://127.0.0.1:8001 --model intent-router --tag vllm_fp16
  python eval_endpoint.py --endpoint http://127.0.0.1:11434 --model intent-q4kM --tag ollama_q4kM
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

sys.path.insert(0, str(Path(__file__).parent))

from baseline_deepseek import evaluate, INTENTS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

SYSTEM_PROMPT = """你是论文资料问答系统的意图分类模块。判断用户问题属于哪一类意图，只输出 JSON。

可选意图：
1. simple_qa：普通问答，问某个事实、概念或结论。
2. metadata_query：元数据查询，问编号、整理者、DOI、附件、缺失情况、清单统计。
3. summary_paper：总结某篇论文，要研究问题、方法、数据、结论。
4. summary_ppt：总结某篇 PPT 或汇报内容。
5. group_report：生成组会汇报提纲。
6. compare_papers：对比多篇文献的异同。

输出格式：{"intent": "六选一"}"""

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_intent(text: str) -> str:
    text = THINK_RE.sub("", text)
    m = re.search(r'"intent"\s*:\s*"([a-z_]+)"', text)
    if m:
        return m.group(1)
    for intent in INTENTS:
        if intent in text:
            return intent
    return "__unparsed__"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True, help="如 http://127.0.0.1:8001")
    parser.add_argument("--model", required=True, help="端点侧模型名，如 intent-router")
    parser.add_argument("--tag", required=True, help="报告命名，如 vllm_fp16 / ollama_q4kM")
    parser.add_argument("--test-file", default=str(DATA_DIR / "splits" / "test.jsonl"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    tests = [json.loads(l) for l in open(args.test_file, encoding="utf-8") if l.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / f"eval_{args.tag}.jsonl"
    report_path = OUT_DIR / f"eval_{args.tag}_report.json"

    done_rows = []
    if raw_path.exists():
        done_rows = [json.loads(l) for l in open(raw_path, encoding="utf-8") if l.strip()]
    todo = tests[len(done_rows):]
    print(f"测试集 {len(tests)} 条，已完成 {len(done_rows)}，本次跑 {len(todo)}")

    url = f"{args.endpoint.rstrip('/')}/v1/chat/completions"
    lock = threading.Lock()
    raw_f = open(raw_path, "a", encoding="utf-8")

    def run_one(item: dict) -> None:
        t0 = time.perf_counter()
        pred = "__error__"
        try:
            resp = requests.post(
                url,
                json={
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": item["question"]},
                    ],
                    "temperature": 0,
                    "max_tokens": 64,
                },
                timeout=120,
            )
            resp.raise_for_status()
            pred = parse_intent(resp.json()["choices"][0]["message"]["content"])
        except Exception as exc:
            pred = f"__error__:{type(exc).__name__}"
        ms = int((time.perf_counter() - t0) * 1000)
        with lock:
            raw_f.write(json.dumps(
                {"question": item["question"], "gold": item["intent"], "pred": pred,
                 "latency_ms": ms}, ensure_ascii=False) + "\n")
            raw_f.flush()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, item) for item in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if i % 60 == 0:
                print(f"  {i}/{len(todo)}，{time.time() - t0:.0f}s")
    raw_f.close()

    rows = [json.loads(l) for l in open(raw_path, encoding="utf-8") if l.strip()]
    bad = [r for r in rows if str(r["pred"]).startswith("__error__") or r["pred"] == "__unparsed__"]
    report = evaluate([r for r in rows if r not in bad])
    report["n_errors_or_unparsed"] = len(bad)
    report["endpoint"] = args.endpoint
    report["model"] = args.model
    report["tag"] = args.tag
    if bad:
        report["error_kinds"] = dict(Counter(r["pred"] for r in bad))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[{args.tag}] accuracy={report['accuracy']}  macro_f1={report['macro_f1']}  "
          f"延迟 avg={report['latency_ms_avg']}ms p50={report['latency_ms_p50']}ms  失败={len(bad)}")
    print(f"报告: {report_path}")


if __name__ == "__main__":
    main()
