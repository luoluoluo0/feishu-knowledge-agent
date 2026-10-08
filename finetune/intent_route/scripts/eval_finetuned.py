"""微调模型正式评测：在冻结测试集上出 accuracy / macro-F1 / 混淆矩阵。

与两个基线完全同构（evaluate 复用 baseline_deepseek），差别只有两点：
- 零样本：微调模型已为此任务训练过，不再给 few-shot 示例；
- 输入是合并后的模型目录（train_sft.py 的 output_dir）。

用法：
  python eval_finetuned.py --model-path /root/autodl-tmp/intent_route/outputs/sft_lr2e4_ep2_r16 --tag lr2e4_ep2_r16
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

from baseline_deepseek import evaluate, INTENTS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

TEST_PATH = DATA_DIR / "splits" / "test.jsonl"
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
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tag", default="finetuned")
    parser.add_argument("--test-file", default=str(TEST_PATH))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    tests = [json.loads(l) for l in open(args.test_file, encoding="utf-8") if l.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / f"eval_{args.tag}.jsonl"
    report_path = OUT_DIR / f"eval_{args.tag}_report.json"

    done = 0
    if raw_path.exists():
        done = sum(1 for l in open(raw_path, encoding="utf-8") if l.strip())
    todo = tests[done:]
    print(f"测试集 {len(tests)} 条，已完成 {done}，本次跑 {len(todo)}")

    print(f"加载模型: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    raw_f = open(raw_path, "a", encoding="utf-8")
    t0 = time.time()
    for i, item in enumerate(todo, 1):
        tt = time.perf_counter()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
            {"question": item["question"], "gold": item["intent"], "pred": parse_intent(text),
             "latency_ms": ms, "raw": text[:200]},
            ensure_ascii=False) + "\n")
        raw_f.flush()
        if i % 30 == 0:
            print(f"  {i}/{len(todo)}，{time.time() - t0:.0f}s")
    raw_f.close()

    rows = [json.loads(l) for l in open(raw_path, encoding="utf-8") if l.strip()]
    errors = [r for r in rows if str(r["pred"]).startswith("__error__") or r["pred"] == "__unparsed__"]
    report = evaluate([r for r in rows if r not in errors])
    report["n_errors_or_unparsed"] = len(errors)
    report["model_path"] = args.model_path
    report["tag"] = args.tag
    if errors:
        report["error_kinds"] = dict(Counter(r["pred"] for r in errors))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[{args.tag}] accuracy={report['accuracy']}  macro_f1={report['macro_f1']}  "
          f"延迟 avg={report['latency_ms_avg']}ms  失败/未解析={len(errors)}")
    print("主要混淆：")
    for c in report["top_confusions"][:5]:
        print(f"  {c['gold']} -> {c['pred']}: {c['count']}")
    print(f"报告: {report_path}")


if __name__ == "__main__":
    main()
