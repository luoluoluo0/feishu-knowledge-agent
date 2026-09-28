"""图表问答评测执行器（方案 A 环节 5）。

对 chart_qa_30.jsonl 逐题打真实系统（/ask SSE），判分三指标：
  通过率   — 事实题/找图题按关键词覆盖；陷阱题按「承认没有」
  图命中   — 来源 sources.image 是否含目标图（找图/事实题）
  幻觉率   — 陷阱题没承认没有、反而给出具体信息的占比

用法：python scripts/run_chart_eval.py [--limit N]
输出：data/eval/result_chart_qa_<时间戳>.jsonl（逐条落盘，可中断）
"""

import argparse
import concurrent.futures
import json
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from app.config import get_settings  # noqa: E402
from app.llm import build_llm  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

EVAL_DIR = PROJECT_DIR / "data" / "eval"
DEFAULT_SET = EVAL_DIR / "chart_qa_30.jsonl"
CAPTIONS_PATH = PROJECT_DIR / "data" / "processed" / "figure_captions.jsonl"
BASE_URL = "http://127.0.0.1:8030"
API_KEY = get_settings().service_api_key

JUDGE_PROMPT = """你是 RAG 评测判分员。判断系统答案是否正确回答了关于图表的问题。

问题：{question}
参考答案：{expected}
图表事实（唯一判分依据，来自 VLM 看图整理）：{claims}

系统答案：{answer}

按顺序应用以下规则，命中即停：
1. 图表事实里**没有**问题问的信息，且答案如实说明「图中未提供/无法确定」→ correct（诚实拒答）
2. 图表事实里**有**该信息，但答案声称「没有/未检索到」→ wrong（证据利用不足，不算诚实拒答）
3. 答案与图表事实矛盾（数值/方位/构成不符，包括采信了论文其他部分的相异信息）→ wrong
4. 答案与图表事实一致且回应了问题 → correct（要点有少量遗漏但主干正确也算 correct）

允许措辞不同，按语义判。只输出 JSON：{{"correct": true/false, "rule": "命中的规则号1-4", "reason": "一句话"}}"""

# 陷阱题的通过词：答案应表达「没有/不确定/未提供」
REFUSAL_MARKS = ["没有", "未提供", "不确定", "无法确定", "未标注", "未明确", "并未", "没有明确", "无法回答", "看不到"]

logger_lock = threading.Lock()


def stream_ask(question: str, thread_id: str, timeout: int = 120) -> dict:
    """打 /ask 的 SSE 流，收完拿 done 事件。与 run_generated_eval 同款。"""

    payload = json.dumps({"question": question, "thread_id": thread_id}).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}/ask/stream",
        data=payload,
        headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
        method="POST",
    )
    done = {}
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[5:])
            except json.JSONDecodeError:
                continue
            if event.get("type") == "done":
                done = event.get("data") or event
    return done


def score_case(case: dict, result: dict) -> dict:
    answer = str(result.get("answer") or "")
    sources = result.get("sources") or []
    images = [str(s.get("image") or "") for s in sources if s.get("image")]
    keywords = case.get("keywords", [])

    record = {
        "case_id": case["case_id"],
        "type": case["type"],
        "question": case["question"],
        "item_id": case.get("item_id"),
        "target_image": case.get("image_path"),
        "answer": answer[:2000],
        "source_images": images,
        "latency_ms": result.get("latency_ms"),
        "token_usage": result.get("token_usage"),
        "error": result.get("error"),
    }

    if case.get("trap"):
        # 幻觉率指标：没承认没有、且答案给了具体信息 → 幻觉
        refused = any(mark in answer for mark in REFUSAL_MARKS)
        record["passed"] = refused
        record["hallucinated"] = not refused
        record["note"] = "拒答" if refused else "疑似硬编（未承认信息缺失）"
        return record

    # 事实/找图题：关键词覆盖过半为过
    hits = [k for k in keywords if k and k in answer]
    record["keywords"] = keywords
    record["keyword_hits"] = hits
    record["keyword_rate"] = round(len(hits) / len(keywords), 2) if keywords else None
    record["passed"] = bool(keywords) and len(hits) / len(keywords) >= 0.5
    # 目标图命中（有目标图的题才计）
    if case.get("image_path"):
        record["figure_hit"] = any(case["image_path"].split("/")[-1] in img for img in images)
    return record


_judge_llm = None


def judge_semantic(case: dict, record: dict, claims_text: str) -> dict | None:
    """语义判分：答案 vs 图表事实的一致性。返回 {correct, reason} 或 None。"""

    global _judge_llm
    if _judge_llm is None:
        _judge_llm = build_llm(get_settings()).bind(temperature=0)
    prompt = JUDGE_PROMPT.format(
        question=case["question"],
        expected=case.get("expected_answer", "")[:300],
        claims=claims_text[:900],
        answer=record.get("answer", "")[:2000],
    )
    try:
        resp = _judge_llm.invoke([SystemMessage(content="只输出 JSON。"), HumanMessage(content=prompt)])
        text = str(resp.content).strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        start, end = text.find("{"), text.rfind("}")
        obj = json.loads(text[start : end + 1])
        return {
            "correct": bool(obj.get("correct")),
            "rule": obj.get("rule"),
            "reason": str(obj.get("reason", ""))[:150],
        }
    except Exception as exc:
        return {"correct": None, "reason": f"judge 失败：{type(exc).__name__}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default=str(DEFAULT_SET))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    cases = [json.loads(l) for l in Path(args.set).read_text(encoding="utf-8").strip().splitlines() if l.strip()]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("评测集为空")
        return 1

    # caption 事实表（语义判分的唯一依据）
    claims_by_image: dict[str, str] = {}
    if CAPTIONS_PATH.exists():
        for line in CAPTIONS_PATH.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            vlm = row.get("vlm") or {}
            claims_by_image[str(row.get("image_path"))] = json.dumps(
                {
                    "图注": row.get("paper_caption"),
                    "事实": vlm.get("key_claims"),
                    "轴": vlm.get("axes"),
                },
                ensure_ascii=False,
            )

    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as resp:
            json.loads(resp.read().decode())
    except Exception as exc:
        print(f"服务预检失败：{exc}")
        return 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EVAL_DIR / f"result_chart_qa_{stamp}.jsonl"
    out_file = out_path.open("w", encoding="utf-8")
    print(f"执行 {len(cases)} 题（并发 {args.concurrency}），结果 → {out_path.name}")

    results: list[dict | None] = [None] * len(cases)
    done_count = 0

    def work(index: int, case: dict):
        nonlocal done_count
        started = time.perf_counter()
        thread_id = f"chartq-{case['case_id'].lower()}-{int(time.time())}"
        try:
            result = stream_ask(case["question"], thread_id)
            result["latency_ms"] = result.get("latency_ms") or int((time.perf_counter() - started) * 1000)
            record = score_case(case, result)
        except Exception as exc:
            record = score_case(case, {})
            record["error"] = f"{type(exc).__name__}: {exc}"[:200]
        # 语义判分（非陷阱题）：答案 vs 图表事实，与关键词口径并列汇报
        if record["type"] != "trap" and not record.get("error"):
            claims = claims_by_image.get(str(case.get("image_path") or ""), "")
            record["judge"] = judge_semantic(case, record, claims)
        results[index] = record
        with logger_lock:
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()
            done_count += 1
            mark = "过" if record.get("passed") else "挂"
            print(f"  [{done_count}/{len(cases)}] {case['case_id']} {mark} {record.get('note','')}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(work, i, c) for i, c in enumerate(cases)]
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as exc:
                print(f"  线程异常: {exc}")
    out_file.close()

    scored = [r for r in results if r]
    traps = [r for r in scored if r["type"] == "trap"]
    facts = [r for r in scored if r["type"] != "trap"]
    fact_pass = sum(1 for r in facts if r.get("passed"))
    trap_pass = sum(1 for r in traps if r.get("passed"))
    fig_cases = [r for r in facts if r.get("figure_hit") is not None]
    fig_hit = sum(1 for r in fig_cases if r.get("figure_hit"))
    hallu = sum(1 for r in traps if r.get("hallucinated"))
    judged = [r for r in facts if (r.get("judge") or {}).get("correct") is not None]
    judge_pass = sum(1 for r in judged if r["judge"]["correct"])
    lat = sorted(r["latency_ms"] for r in scored if r.get("latency_ms"))

    print("\n" + "=" * 60)
    print(f"事实/找图题：关键词通过 {fact_pass}/{len(facts)}；语义判分通过 {judge_pass}/{len(judged)}")
    print(f"陷阱题拒答：{trap_pass}/{len(traps)}（幻觉 {hallu} 条）")
    print(f"目标图命中：{fig_hit}/{len(fig_cases)}")
    if lat:
        print(f"延迟中位 {lat[len(lat)//2]/1000:.1f}s")
    print(f"已写入 {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
