"""LLM-as-Judge：离线评判已落盘的 e2e 答案。

为什么需要：关键词判分是字面匹配——答案换了个说法（「可行能力」写成
「capability」）就判错，其实是对的；反过来，硬凑出来的关键词也可能掩盖
编造。judge 拿评测集里的 reference_answer 做语义锚点，判两件事：

- 忠实度：回答是否覆盖参考答案的核心要点。允许同义改写、详略不同、
  中英文对照，只看事实是否一致；
- 编造：回答是否编造了参考答案撑不住的具体信息（不存在的编号、DOI、
  作者、数字、结论）。

refusal 类单独判：这类题在语料里没有答案，正确行为是承认查不到——
硬凑文献就是失败。

关键设计：**不重新提问**。e2e 跑评测时答案已经存进结果文件，judge 只读
文件、批量打分，几十秒跑完一遍，改判据后重跑成本几乎为零。
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_DIR / "data" / "eval"

# 判分模型温度必须是 0：judge 要的是可复现的一致判定，不是创造性。
JUDGE_TEMPERATURE = 0.0
DEFAULT_CONCURRENCY = 6

_FAITHFULNESS_RULES = """判定规则：
- "faithful"：系统回答是否在语义上覆盖了参考答案的核心要点。允许同义改写、
  详略不同、中英文对照（英文要点用中文答出算对）；只看事实是否一致。
- "fabricated"：系统回答的具体信息是否与参考答案**矛盾**（编号、作者、数字、
  结论对不上、张冠李戴），或明显凭空捏造关键内容。注意：回答比参考答案
  **更详细**（多出编号、DOI、作者、页码、数字等）本身不算编造——这些细节
  应推定来自系统检索到的文献，正是 RAG 该做的；只有当它们与参考答案冲突
  或明显荒谬时才判 true。

只输出一个 JSON 对象，不要输出任何其他内容：
{"faithful": true或false, "fabricated": true或false, "reason": "一句话依据"}"""

_REFUSAL_RULES = """这道题在文献库里没有答案。系统的正确行为是明确承认查不到、
说明语料不覆盖，而不是硬凑一篇看起来相关的文献。

只输出一个 JSON 对象，不要输出任何其他内容：
{"appropriate": true或false, "reason": "一句话依据"}"""


def load_case_map() -> dict[str, dict]:
    """评测集按 id 建索引（结果文件里只有 id，没有题目原文）。"""

    cases: dict[str, dict] = {}
    for name in ("eval_set.jsonl", "eval_special.jsonl"):
        path = EVAL_DIR / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("id"):
                    cases[str(row["id"])] = row
    return cases


def _question_of(case: dict) -> str:
    """多轮题取最后一轮的追问当问题（前几轮是历史铺垫）。"""

    turns = case.get("turns")
    if isinstance(turns, list) and turns:
        last = turns[-1]
        if isinstance(last, dict) and last.get("content"):
            return str(last["content"])
    return str(case.get("query", ""))


def _build_prompt(case: dict, answer: str) -> str:
    if case.get("expected_refusal"):
        return (
            f"【问题】{_question_of(case)}\n\n"
            f"【系统回答】\n{answer}\n\n{_REFUSAL_RULES}"
        )
    reference = str(case.get("reference_answer", "")).strip() or "（无参考答案，按常识判断事实一致性）"
    return (
        f"【问题】{_question_of(case)}\n\n"
        f"【参考答案】\n{reference}\n\n"
        f"【系统回答】\n{answer}\n\n"
        f"{_FAITHFULNESS_RULES}"
    )


def _extract_json(text: str) -> dict | None:
    """模型偶尔会在 JSON 前后带话，抠出第一个完整 JSON 对象。"""

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _judge_one(llm: ChatOpenAI, case: dict, answer: str) -> dict:
    """评一条。返回统一结构：{ok, kind, fabricated, reason}。"""

    kind = "refusal" if case.get("expected_refusal") else "faithfulness"
    prompt = _build_prompt(case, answer)
    verdict: dict | None = None
    # 失败原因要带出去：余额不足（402）、限流、JSON 解析失败是三种完全
    # 不同的问题，混成一句「调用失败」就没法排查（2026-09-21 余额耗尽
    # 95 条全 0，原因在文件里却看不出是 402，教训）。
    last_error = "judge 调用失败（未知原因）"
    for attempt in range(2):
        try:
            resp = llm.invoke(
                [
                    SystemMessage(content="你是严格的 RAG 评测评审，只输出 JSON。"),
                    HumanMessage(content=prompt),
                ]
            )
        except Exception as exc:
            verdict = None
            last_error = f"调用失败：{type(exc).__name__}: {exc}"
            logger.warning("judge 调用失败（第 %d 次）：%s", attempt + 1, exc)
            continue
        verdict = _extract_json(str(resp.content))
        if verdict is None:
            last_error = "输出无法解析为 JSON：" + str(resp.content)[:100]
    if verdict is None:
        return {"ok": False, "kind": kind, "fabricated": False, "reason": last_error[:180]}

    reason = str(verdict.get("reason", ""))[:200]
    if kind == "refusal":
        appropriate = bool(verdict.get("appropriate"))
        return {"ok": appropriate, "kind": kind, "fabricated": not appropriate, "reason": reason}
    faithful = bool(verdict.get("faithful"))
    fabricated = bool(verdict.get("fabricated"))
    # 不忠实但也没编造（漏答要点）不算 ok，但和编造要分开统计——
    # 漏答是检索/生成的召回问题，编造是幻觉问题，治理手段不同。
    return {"ok": faithful and not fabricated, "kind": kind, "fabricated": fabricated, "reason": reason}


def run_judge(
    result_path: Path,
    out_path: Path | None = None,
    limit: int | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    on_progress=None,
) -> dict:
    """评判一份 e2e 结果文件，落盘 judged_*.jsonl，返回汇总。"""

    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=JUDGE_TEMPERATURE,
        timeout=settings.llm_timeout_seconds,
        max_retries=2,
    )
    cases = load_case_map()

    rows: list[dict] = []
    with result_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit is not None:
        rows = rows[: max(0, limit)]

    judged_at = datetime.now().isoformat(timespec="seconds")
    results: list[dict] = [None] * len(rows)  # type: ignore[list-item]
    done_count = 0
    total = len(rows)
    if on_progress:
        on_progress(0, total)

    def _work(index: int, row: dict) -> None:
        nonlocal done_count
        case = cases.get(str(row.get("id", "")))
        answer = str(row.get("answer") or "").strip()
        if case is None:
            verdict = {"ok": False, "kind": "faithfulness", "fabricated": False, "reason": "评测集中找不到该 id"}
        elif not answer:
            verdict = {"ok": False, "kind": "refusal" if (case or {}).get("expected_refusal") else "faithfulness",
                       "fabricated": False, "reason": "系统没有返回回答"}
        else:
            verdict = _judge_one(llm, case, answer)
        results[index] = {
            "id": row.get("id"),
            "category": row.get("category"),
            "judged_at": judged_at,
            **verdict,
        }
        done_count += 1
        if on_progress:
            on_progress(done_count, total)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_work, i, row) for i, row in enumerate(rows)]
        for future in futures:
            future.result()

    if out_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = EVAL_DIR / f"judged_{stamp}.jsonl"
    meta = {
        "_meta": {
            "source": result_path.name,
            "judge_model": settings.deepseek_model,
            "finished_at": judged_at,
            "total": total,
        }
    }
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return summarize_rows(results, meta["_meta"])  # type: ignore[arg-type]


def summarize_rows(rows: list[dict], meta: dict) -> dict:
    """把 judged 行汇总成报告要的形状（api 端点读文件后也走这里）。"""

    by_category: dict[str, dict] = {}
    for row in rows:
        cat = str(row.get("category") or "?")
        s = by_category.setdefault(cat, {"n": 0, "ok": 0, "fabricated": 0})
        s["n"] += 1
        if row.get("ok"):
            s["ok"] += 1
        if row.get("fabricated"):
            s["fabricated"] += 1
    for s in by_category.values():
        s["ok_rate"] = round(s["ok"] / s["n"], 4) if s["n"] else None
    ok_total = sum(1 for r in rows if r.get("ok"))
    fabricated_total = sum(1 for r in rows if r.get("fabricated"))
    return {
        "source": meta.get("source"),
        "judge_model": meta.get("judge_model"),
        "finished_at": meta.get("finished_at"),
        "total": len(rows),
        "ok_rate": round(ok_total / len(rows), 4) if rows else None,
        "fabricated": fabricated_total,
        "by_category": by_category,
    }


def latest_judged() -> dict | None:
    """读最新一份 judged 文件并汇总；还没有就返回 None。"""

    candidates = sorted(EVAL_DIR.glob("judged_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    path = candidates[-1]
    meta: dict = {}
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in row:
                meta = row["_meta"]
            else:
                rows.append(row)
    if not meta:
        return None
    summary = summarize_rows(rows, meta)
    summary["file"] = path.name
    return summary
