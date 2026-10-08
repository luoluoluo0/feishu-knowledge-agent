"""基线二（transformers 版）：Qwen3-1.7B 基座 few-shot，零训练对照。

不依赖 vLLM/任何服务——直接 transformers 加载权重推理，设计在
AutoDL 训练环境里跑（unsloth 安装自带 transformers），本机 8G 卡
也能跑（1.7B FP16 约 3.4G 显存）。

与 baseline_deepseek.py 同一份 test、同一套指标（evaluate 复用）。
few-shot 示例只从 train 取（每类 1 条最短真实种子），不碰 test/val。

用法（AutoDL 上，先上传 splits/ 和本脚本）：
  python baseline_qwen_fewshot_hf.py --model-path /root/autodl-tmp/models/Qwen3-1.7B
  python baseline_qwen_fewshot_hf.py --model-path Qwen/Qwen3-1.7B   # 从 HF 缓存/在线
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT))

from baseline_deepseek import evaluate, INTENTS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
TEST_PATH = DATA_DIR / "splits" / "test.jsonl"
TRAIN_PATH = DATA_DIR / "splits" / "train.jsonl"
RAW_PATH = OUT_DIR / "baseline_qwen_fewshot_hf.jsonl"
REPORT_PATH = OUT_DIR / "baseline_qwen_fewshot_hf_report.json"

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


def parse_intent(text: str) -> str:
    text = THINK_RE.sub("", text)
    m = re.search(r'"intent"\s*:\s*"([a-z_]+)"', text)
    if m:
        return m.group(1)
    for intent in INTENTS:
        if intent in text:
            return intent
    return "__unparsed__"


def pick_few_shot(train_rows: list[dict]) -> list[dict]:
    by_intent: dict[str, list] = {}
    for r in train_rows:
        if r.get("source") == "distill":
            continue
        by_intent.setdefault(r["intent"], []).append(r)
    messages = []
    for intent in INTENTS:
        rows = by_intent.get(intent) or []
        if rows:
            ex = min(rows, key=lambda r: len(r["question"]))
            messages.append({"role": "user", "content": ex["question"]})
            messages.append({"role": "assistant",
                             "content": json.dumps({"intent": intent, "reason": "示例"}, ensure_ascii=False)})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/root/autodl-tmp/models/Qwen3-1.7B")
    parser.add_argument("--max-new-tokens", type=int, default=200)
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

    print(f"加载模型: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    def build_prompt(question: str) -> str:
        messages = [{"role": "system", "content": SYSTEM}, *few_shot,
                    {"role": "user", "content": question}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:  # 模板不支持 enable_thinking 参数时退回默认
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    raw_f = open(RAW_PATH, "a", encoding="utf-8")
    t0 = time.time()
    for i, item in enumerate(todo, 1):
        tt = time.perf_counter()
        prompt = build_prompt(item["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        ms = int((time.perf_counter() - tt) * 1000)
        raw_f.write(json.dumps(
            {"question": item["question"], "gold": item["intent"],
             "pred": parse_intent(text), "latency_ms": ms},
            ensure_ascii=False) + "\n")
        raw_f.flush()
        if i % 20 == 0:
            print(f"  {i}/{len(todo)}，{time.time() - t0:.0f}s")
    raw_f.close()

    rows = [json.loads(l) for l in open(RAW_PATH, encoding="utf-8") if l.strip()]
    errors = [r for r in rows if str(r["pred"]).startswith("__error__") or r["pred"] == "__unparsed__"]
    report = evaluate([r for r in rows if r not in errors])
    report["n_errors_or_unparsed"] = len(errors)
    if errors:
        report["error_kinds"] = dict(Counter(r["pred"] for r in errors))
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\naccuracy={report['accuracy']}  macro_f1={report['macro_f1']}  "
          f"延迟 avg={report['latency_ms_avg']}ms  失败/未解析={len(errors)}")
    print("主要混淆：")
    for c in report["top_confusions"][:5]:
        print(f"  {c['gold']} -> {c['pred']}: {c['count']}")
    print(f"报告: {REPORT_PATH}")


if __name__ == "__main__":
    main()
