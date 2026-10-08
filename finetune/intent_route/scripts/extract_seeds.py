"""从三个来源提取 (query, intent) 种子对。

来源与优先级（同题重复时保留优先级高的）：
1. eval_main : data/eval/eval_set.jsonl 的 300 条主集，category 经 CATEGORY_INTENT
   映射成意图（映射从 scripts/run_eval.py 抄来，refusal/coreference/edge 按设计
   不设意图，跳过）；
2. trap      : data/eval/generated_set_100.jsonl 联合 result_gen_20260921_201001.jsonl，
   只取 intent_ok=True 的行（金标准）；intent_ok=False 的写去 trap_review.csv 人工复核；
3. prod      : data/runtime/conversation_history.db 的 conversation_turns 表，
   intent 非空的行。

输出：
- finetune/intent_route/data/seeds.jsonl      全部种子
- finetune/intent_route/data/trap_review.csv  待人工复核的陷阱题
- 终端打印类别分布 + 写 outputs/seeds_distribution.txt

去重口径：去空格与中英文标点后取哈希，与 docs/project2-finetune-plan.md 的
Week 1 规格一致。
"""

import csv
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "data" / "eval"
OUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# category -> 期望意图。与 scripts/run_eval.py 的 CATEGORY_INTENT 保持一致；
# 那边的注释解释了为什么 refusal/coreference 不设意图（拒答是结果不是意图、
# 多轮题的意图看最后一轮）。
CATEGORY_INTENT = {
    "metadata": "metadata_query",
    "summary_paper": "summary_paper",
    "compare_papers": "compare_papers",
    "group_report": "group_report",
    "en_fact": "simple_qa",
    "zh_fact": "simple_qa",
    "cross_lingual": "simple_qa",
}

NORMALIZE_RE = re.compile(r"[\s，。？！、；：""''「」（）,.?!;:'\"()]")


def normalize(text: str) -> str:
    return NORMALIZE_RE.sub("", (text or "").strip())


def question_hash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    seeds: list[dict] = []
    seen: set[str] = set()
    stats = Counter()

    def add(question: str, intent: str, source: str, source_id: str, note: str = "") -> bool:
        h = question_hash(question)
        if not question.strip() or not intent or h in seen:
            stats[f"dup_or_skip:{source}"] += 1
            return False
        seen.add(h)
        seeds.append(
            {
                "question": question.strip(),
                "intent": intent,
                "source": source,
                "source_id": source_id,
                "note": note,
            }
        )
        stats[f"kept:{source}"] += 1
        return True

    # ---- 来源 1：300 条主集，category 映射意图 ----
    for row in load_jsonl(EVAL_DIR / "eval_set.jsonl"):
        intent = CATEGORY_INTENT.get(row.get("category", ""))
        add(row.get("query", ""), intent, "eval_main", row.get("id", ""), row.get("category", ""))

    # ---- 来源 2：陷阱集，只收 intent_ok=True；False 的送去人工复核 ----
    traps = {row["case_id"]: row for row in load_jsonl(EVAL_DIR / "generated_set_100.jsonl")}
    review_rows = []
    for row in load_jsonl(EVAL_DIR / "result_gen_20260921_201001.jsonl"):
        case = traps.get(row.get("case_id"))
        if not case or row.get("error"):
            continue
        question, intent = case.get("question", ""), row.get("intent", "")
        if row.get("intent_ok"):
            add(question, intent, "trap", row.get("case_id", ""), case.get("trap_type", ""))
        elif intent:
            review_rows.append(
                {"case_id": row["case_id"], "question": question, "intent": intent,
                 "trap_type": case.get("trap_type", "")}
            )

    # ---- 来源 3：生产日志 ----
    db_path = ROOT / "data" / "runtime" / "conversation_history.db"
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT question, intent FROM conversation_turns "
            "WHERE intent IS NOT NULL AND intent != ''"
        ).fetchall()
    finally:
        con.close()
    for i, (question, intent) in enumerate(rows):
        add(question, intent, "prod", f"turn_{i}")

    # ---- 落盘 ----
    seeds_path = OUT_DIR / "seeds.jsonl"
    with open(seeds_path, "w", encoding="utf-8") as f:
        for seed in seeds:
            f.write(json.dumps(seed, ensure_ascii=False) + "\n")

    review_path = OUT_DIR / "trap_review.csv"
    with open(review_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "question", "intent", "trap_type"])
        writer.writeheader()
        writer.writerows(review_rows)

    # ---- 分布报告 ----
    by_intent = Counter(s["intent"] for s in seeds)
    by_source = Counter(s["source"] for s in seeds)
    lines = [
        f"种子总数: {len(seeds)}",
        f"来源分布: {dict(by_source)}",
        "意图分布:",
    ]
    lines += [f"  {intent}: {n}" for intent, n in by_intent.most_common()]
    lines += [
        f"重复/跳过: {sum(v for k, v in stats.items() if k.startswith('dup'))}",
        f"待人工复核陷阱题: {len(review_rows)} -> {review_path.name}",
    ]
    report = "\n".join(lines)
    print(report)
    report_path = Path(__file__).resolve().parents[1] / "outputs" / "seeds_distribution.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
