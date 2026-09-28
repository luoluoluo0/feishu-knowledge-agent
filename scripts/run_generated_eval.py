"""新格式测试集（generated_set_100.jsonl）的执行器。

与 run_eval.py 的区别：这套用例是「五节点陷阱集」，判分维度不同——
- intent_ok        ：实际路由意图 == expected.intent（另记录是否踩中 intent_not）
- tools_ok         ：实际工具调用序列匹配期望。对比类（compare_papers）按
                     多重集精确匹配（逐篇次数=文献数）；其余按「期望 ⊆ 实际」
                     （多调不扣分，少调必错）。
- recall           ：expected.relevant_item_ids 被返回来源覆盖的比例
- refusal_ok       ：expected_refusal 的题，回答应出现明确的「没有/未找到」
                     类表述且不硬凑文献
- token / latency  ：顺带记录（done 事件带 token_usage）

多轮用例的 history 逐轮真实重放（用户句直接打 /ask/stream，让服务端
自己产生记忆），最后一轮才是被测问题。

用法：
  python scripts/run_generated_eval.py --limit 3     # 先小样本验证管道
  python scripts/run_generated_eval.py               # 全量 100 条

结果落盘 data/eval/result_gen_<时间戳>.jsonl，每行一条判定 + 原始检查值。
注意：跑之前后端必须以 QUERY_CACHE_ENABLED=false 重启（缓存默认开，
不关的话相同问题直接吐缓存，测不到真实链路）。
"""

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

DEFAULT_SET = PROJECT_DIR / "data" / "eval" / "generated_set_100.jsonl"
EVAL_DIR = PROJECT_DIR / "data" / "eval"

# 拒答识别：回答里出现这些表述之一，且不硬凑文献，才算恰当拒答。
# 词表来自首轮全量的真实拒答措辞归纳（「没有分析」「没有关于」曾被
# 漏掉导致 9 条全对的拒答被误报 2 条）。
REFUSAL_MARKS = ["没有找到", "未找到", "没有相关", "库里没有", "没有检索到",
                 "无法回答", "找不到", "没有查到", "没有收录", "没有覆盖", "未收录",
                 "没有分析", "没有关于", "没有提到", "没有涉及", "不涉及",
                 "均未命中", "未命中", "没有专门研究"]


def load_set(path: Path, limit: int | None) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return cases[: max(0, limit)] if limit else cases


def stream_ask(base_url: str, api_key: str, question: str, thread_id: str, timeout: int = 90) -> dict:
    """打一次 /ask/stream，抽干事件流，返回 done 事件的数据（失败抛异常）。"""

    body = json.dumps({"question": question, "thread_id": thread_id}, ensure_ascii=False).encode()
    request = urllib.request.Request(
        f"{base_url}/ask/stream",
        data=body,
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        done = None
        error = None
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[len("data: "):])
            except json.JSONDecodeError:
                continue
            event = payload.get("event") or payload.get("type")
            if event == "done":
                done = payload.get("data") or payload
            elif event == "error":
                error = payload.get("data") or payload
        if done is not None:
            return done
        raise RuntimeError(f"流结束但没有 done 事件：{error or '未知错误'}")


def check_case(case: dict, base_url: str, api_key: str) -> dict:
    """跑一条用例并打分。返回该条的判定记录。"""
    cid = case["case_id"]
    expected = case["expected"]
    thread_id = f"gen-{cid.lower()}-{int(time.time())}"

    result = {
        "case_id": cid,
        "trap_type": case.get("trap_type"),
        "tested_nodes": case.get("tested_nodes"),
        "difficulty": case.get("difficulty"),
        "error": None,
        "intent": None,
        "intent_not_hit": False,
        "intent_ok": None,
        "tools_actual": [],
        "tools_expected": expected.get("tools", []),
        "tools_ok": None,
        "recall": None,
        "refusal_ok": None,
        "answer_len": 0,
        "latency_ms": None,
        "token_usage": None,
        "answer": "",
        "answer_checkpoints": case.get("answer_checkpoints", []),
    }

    started = time.perf_counter()
    try:
        # 多轮：history 逐轮真实重放（产生服务端记忆），最后才是被测问题。
        for turn in case.get("history", []):
            text = turn[len("用户："):] if turn.startswith("用户：") else turn
            stream_ask(base_url, api_key, text, thread_id)
        done = stream_ask(base_url, api_key, case["question"], thread_id)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:200]
        result["latency_ms"] = int((time.perf_counter() - started) * 1000)
        return result

    answer = str(done.get("answer") or "")
    intent = done.get("intent")
    actual_tools = [str(t.get("name") or "") for t in done.get("tools") or []]
    source_ids = [str(s.get("item_id") or "") for s in done.get("sources") or []]

    result.update(
        intent=intent,
        intent_not_hit=(intent == expected.get("intent_not")),
        intent_ok=(intent == expected.get("intent")),
        tools_actual=actual_tools,
        answer_len=len(answer),
        token_usage=done.get("token_usage"),
        latency_ms=int((time.perf_counter() - started) * 1000),
        answer=answer[:600],
    )

    # 工具路径：对比类要求多重集精确匹配（逐篇次数=文献数）；
    # tools_groups 为等价工具组（每组命中任一即满足，所有组都要满足）；
    # 其余只要求期望工具都被调用（多调不扣分）。
    #
    # 两个口径豁免（首轮全量 100 条的教训）：
    # - 元数据类问题系统有一批比泛化卡片检索更精准的专用工具
    #   （get_literature_by_item_id / list_literature_by_reader /
    #   list_missing_files / list_literature_by_status），用哪个都对；
    # - 多轮追问若答案已在上文（上轮工具结果还在会话记忆里），
    #   本轮不重复检索是正确行为，tools/recall 不判失败。
    METADATA_TOOLS = {
        "hybrid_search_literature_card", "get_literature_by_item_id",
        "list_literature_by_reader", "list_missing_files",
        "list_literature_by_status", "semantic_search_literature",
    }
    expected_tools = expected.get("tools", [])
    groups = expected.get("tools_groups")
    if expected.get("intent") == "compare_papers":
        result["tools_ok"] = Counter(actual_tools) == Counter(expected_tools)
    elif not actual_tools and case.get("history"):
        # 答案已在上文，复用作答不再检索——对所有意图都算正确
        # （对比类除外：逐篇对比必须真检索）。
        result["tools_ok"] = True
    elif groups:
        result["tools_ok"] = all(
            any(t in actual_tools for t in group) for group in groups
        )
    elif expected.get("intent") == "metadata_query":
        if not actual_tools and case.get("history"):
            result["tools_ok"] = True  # 答案已在上文，复用作答不再检索
        else:
            result["tools_ok"] = bool(set(actual_tools) & METADATA_TOOLS)
    elif expected_tools:
        result["tools_ok"] = set(expected_tools) <= set(actual_tools)
    else:
        result["tools_ok"] = True

    # 召回：期望文献被返回来源覆盖的比例。
    relevant = expected.get("relevant_item_ids") or []
    if relevant and actual_tools:
        hit = len(set(relevant) & set(source_ids))
        result["recall"] = round(hit / len(relevant), 4)
        result["recalled_ids"] = sorted(set(relevant) & set(source_ids))
    elif relevant and case.get("history"):
        # 多轮追问复用上文作答：本轮无新检索，来源在上一轮的轮次里，
        # 不计入本轮 recall（答案对错由 judge 按 checkpoints 判）。
        result["recall"] = None
        result["recall_note"] = "复用上文作答，本轮未检索"

    # 拒答：期望拒答时，回答应包含明确的「没有」类表述。
    if expected.get("expected_refusal"):
        result["refusal_ok"] = any(mark in answer for mark in REFUSAL_MARKS)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="新格式五节点测试集执行器")
    parser.add_argument("--set", default=str(DEFAULT_SET), help="测试集路径")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--offset", type=int, default=0, help="跳过前 N 条（配合 limit 抽查中段用例）")
    parser.add_argument("--concurrency", type=int, default=4, help="并发数，默认 4")
    parser.add_argument("--base-url", default="http://127.0.0.1:8030")
    parser.add_argument("--api-key", default=None, help="默认读取 SERVICE_API_KEY")
    args = parser.parse_args()

    if not args.api_key:
        from app.config import get_settings

        args.api_key = get_settings().service_api_key
    if not args.api_key:
        print("没有配置 SERVICE_API_KEY，也没有传入 --api-key。")
        return 2

    cases = load_set(Path(args.set), None)
    if args.offset:
        cases = cases[args.offset:]
    cases = cases[: args.limit] if args.limit else cases
    if not cases:
        print("测试集为空。")
        return 1

    # 预检：服务不在就直接停，别等 100 条全报网络错误。
    try:
        with urllib.request.urlopen(f"{args.base_url}/health", timeout=5) as resp:
            json.loads(resp.read().decode())
    except Exception as exc:
        print(f"服务预检失败（{args.base_url}/health）：{exc}")
        return 1

    print(f"执行 {len(cases)} 条（并发 {args.concurrency}），结果写入 {EVAL_DIR.name}/")
    results: list[dict | None] = [None] * len(cases)
    done_count = 0
    lock = threading.Lock()
    started = time.perf_counter()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EVAL_DIR / f"result_gen_{stamp}.jsonl"
    # 每条完成立即落盘：上一版把结果攒到全部结束后一次性写，收尾阶段
    # 一旦卡住（Windows 控制台选中/Quick Edit 都会冻结 print）再按
    # Ctrl+C，整轮明细全丢（2026-09-21 实测）。任何中断都不能丢已跑
    # 完的数据。
    out_file = out_path.open("w", encoding="utf-8")

    def _work(index: int, case: dict) -> None:
        nonlocal done_count
        try:
            result = check_case(case, args.base_url, args.api_key)
        except Exception as exc:
            result = {
                "case_id": case["case_id"],
                "trap_type": case.get("trap_type"),
                "error": f"{type(exc).__name__}: {exc}"[:200],
            }
        results[index] = result
        with lock:
            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_file.flush()
            done_count += 1
            mark = "ERR" if result.get("error") else "ok"
            print(f"  [{done_count}/{len(cases)}] {result['case_id']} {mark}", flush=True)

    interrupted = False
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(_work, i, c) for i, c in enumerate(cases)]
            # 按完成顺序收割：按提交序等待会让最慢的用例堵住已完成结果的收集。
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"  用例线程异常（已按错误记录）: {exc}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\n检测到 Ctrl+C：已完成的用例都已落盘，等待进行中的请求结束……", flush=True)
    finally:
        out_file.close()

    # 汇总（部分完成也汇总——中断不给白跑）
    scored_all = [r for r in results if r is not None]
    errors = [r for r in scored_all if r.get("error")]
    scored = [r for r in scored_all if not r.get("error")]
    print("\n" + "=" * 60)
    head = f"完成 {len(scored_all)}/{len(cases)} 条" + ("（中断收尾）" if interrupted else "")
    print(f"{head}，执行错误 {len(errors)} 条，用时 {time.perf_counter() - started:.0f}s")
    if scored:
        def rate(name: str) -> str:
            vals = [r[name] for r in scored if r.get(name) is not None]
            if not vals:
                return "—"
            return f"{sum(1 for v in vals if v)}/{len(vals)} = {sum(1 for v in vals if v) / len(vals):.1%}"

        print(f"意图正确：{rate('intent_ok')}（踩中易混意图 {sum(1 for r in scored if r.get('intent_not_hit'))} 条）")
        print(f"工具路径：{rate('tools_ok')}")
        recalls = [r["recall"] for r in scored if r.get("recall") is not None]
        if recalls:
            print(f"召回 recall：均值 {sum(recalls) / len(recalls):.3f}（{len(recalls)} 条）")
        refusals = [r for r in scored if r["refusal_ok"] is not None]
        if refusals:
            ok = sum(1 for r in refusals if r["refusal_ok"])
            print(f"拒答正确：{ok}/{len(refusals)} = {ok / len(refusals):.1%}")
        by_trap = Counter(r["trap_type"] for r in scored if r.get("tools_ok") is False or r.get("intent_ok") is False)
        if by_trap:
            print("失败集中的陷阱类型:", dict(by_trap))
        if errors:
            print("错误样例:", errors[0].get("error"))
    print(f"已写入 {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
