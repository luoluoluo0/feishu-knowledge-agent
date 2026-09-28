import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

try:
    from app.config import get_settings
    from app.eval_scoring import score_case
except ModuleNotFoundError as exc:
    print(f"缺少依赖：{exc.name}")
    print("先 conda activate base 再跑。")
    raise SystemExit(1)


# 拿测试集去跑，看系统实际表现。
#
# 两种模式，测的东西不一样，都要跑：
#
#   retrieval —— 只跑检索这一层，不调大模型生成答案。
#                快（每条几十毫秒），判分硬：期望的块在不在 top-k 里。
#                检索坏了但生成把话说圆了，只有这个模式看得出来。
#
#   e2e       —— 打 HTTP /ask，走完整链路。
#                慢（每条几十秒），判分软：关键事实词覆盖了多少、
#                该拒答的有没有硬凑、意图选对没有、指代解对没有。
#                生成坏了但检索是对的，只有这个模式看得出来。
#
# 多轮指代题在 retrieval 模式下跳过——指代是对话现象，
# 单次检索里没有「上一轮」这回事。e2e 模式会重放前两轮再判第三轮。


# 每类期望走哪条链路。用来算意图识别准确率。
# edge 类不设期望——那种输入本来就无意图可言。
CATEGORY_INTENT = {
    "metadata": "metadata_query",
    "summary_paper": "summary_paper",
    "compare_papers": "compare_papers",
    "group_report": "group_report",
    "en_fact": "simple_qa",
    "zh_fact": "simple_qa",
    "cross_lingual": "simple_qa",
    # refusal 和 coreference 不设期望意图。
    #
    # 拒答是「结果」，不是「意图」——用户会以任何意图问出一个库里没有
    # 答案的问题。实测模板拒答句「这个文献库里有没有关于 X 的论文」
    # 被正确分成了 metadata_query，早期却标成 simple_qa。
    #
    # 多轮题同理：第三轮问什么，意图就是什么。实测「那篇文献里，为什么
    # 需要进行额外的测试？」被分成 summary_paper 是对的，标成
    # simple_qa 属于硬套。两处标注错误合计白扣了 24 条准确率。
}

PLANNER_INTENTS = {"group_report", "compare_papers"}


def load_cases(path: Path, categories: list[str] | None, limit: int) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if categories and case.get("category") not in categories:
                continue
            cases.append(case)
    if limit:
        cases = cases[:limit]
    return cases


def load_corpus_titles() -> tuple[list[str], dict[str, str]]:
    """语料里的文献标题。

    返回 (标题列表, item_id → 标题)。前者给拒答判定用；后者给跨语言题
    判分用——跨语言题要看回复里有没有引用期望的那一篇。
    """

    chunks = PROJECT_DIR / "data" / "processed" / "chunks_v2.jsonl"
    titles: set[str] = set()
    title_map: dict[str, str] = {}
    if not chunks.exists():
        return [], {}

    with chunks.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            title = (chunk.get("title") or "").strip()
            item_id = str(chunk.get("item_id") or "")
            if len(title) >= 8:
                titles.add(title)
            if item_id and title and item_id not in title_map:
                title_map[item_id] = title

    return sorted(titles), title_map


# ---------------------------------------------------------------- 检索模式


def run_retrieval(cases: list[dict], args) -> list[dict]:
    from app.milvus_store import MilvusStore

    settings = get_settings()
    store = MilvusStore(settings, collection=args.collection or None)

    if not store.has_collection():
        print(f"集合不存在：{store.collection}")
        return []

    # 跑混合检索前先确认这个集合真有 BM25。
    #
    # 没有的话，每一条都会以 Milvus 的「failed to get field schema by name:
    # fieldName(sparse)」告终——300 条报 279 条，看的人以为是检索坏了，
    # 其实只是集合选错了。配置默认指向旧集合，很容易踩。
    described = store.client.describe_collection(store.collection)
    field_names = {f["name"] for f in described.get("fields", [])}
    has_bm25 = "sparse" in field_names and bool(described.get("functions"))

    if args.retrieval_mode in ("rrf", "compare") and not has_bm25:
        print(f"集合 {store.collection} 没有 BM25（缺 sparse 字段或 BM25 函数）。")
        print("跑混合检索会逐条报错，不是检索坏了，是集合选错了。")
        print()
        print("有 BM25 的集合：")
        for name in store.client.list_collections():
            info = store.client.describe_collection(name)
            names = {f["name"] for f in info.get("fields", [])}
            if "sparse" in names and info.get("functions"):
                print(f"  {name}")
        print()
        print("加上参数重跑：--collection <上面那个>")
        return []

    print(f"Milvus  ：{settings.milvus_uri}")
    print(f"集合    ：{store.collection}    条数：{store.count()}")
    print(f"检索方式：{args.retrieval_mode}    top-k：{args.top_k}")
    print()

    def one(case: dict) -> dict:
        case_id = case["id"]
        category = case["category"]

        # 指代题单次检索测不了：没有「上一轮」这回事。
        if case.get("turns"):
            return {
                "id": case_id,
                "category": category,
                "skipped": True,
                "reason": "多轮题在检索模式下跳过，用 --mode e2e 测。",
            }

        started = time.perf_counter()
        results = None
        error = None
        try:
            if args.retrieval_mode == "dense":
                results = store.search(case["query"], top_k=args.top_k)
            else:
                results = store.hybrid_search(case["query"], top_k=args.top_k)
        except Exception as exc:
            # 截到 200 字：Milvus 的报错把真正的原因放在后半句
            # （「failed to create query plan」后面才是根因），
            # 截太短只剩前半句，看不出问题在哪。
            error = f"{type(exc).__name__}: {str(exc)[:200]}"

        latency = (time.perf_counter() - started) * 1000
        score = score_case(case, results=results, error=error)
        return {
            "id": case_id,
            "category": category,
            "passed": score.passed,
            "unjudged": score.unjudged,
            "checks": score.checks,
            "latency_ms": latency,
            "error": error,
        }

    return run_batch(cases, one, args.workers)


# ---------------------------------------------------------------- 端到端模式


def post_ask(base_url: str, api_key: str, question: str, thread_id: str, timeout: int) -> dict:
    body = json.dumps(
        {"question": question, "thread_id": thread_id, "include_trace": False}
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/ask",
        data=body,
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def run_e2e(
    cases: list[dict], args, titles: list[str], title_map: dict[str, str]
) -> list[dict]:
    settings = get_settings()
    api_key = settings.service_api_key

    if not api_key:
        print("服务端没有配置 FEISHU_SERVICE_API_KEY，无法调用 /ask。")
        return []

    try:
        with urllib.request.urlopen(f"{args.base_url}/health", timeout=10) as response:
            health = json.loads(response.read().decode("utf-8"))
        print(f"服务    ：{args.base_url}    状态：{health.get('status', '?')}")
    except Exception as exc:
        print(f"连不上服务 {args.base_url}：{exc}")
        print()
        print("先启动 API：")
        print("  python scripts/run_api.py")
        return []

    print(f"超时    ：{args.timeout} 秒/条    并发：{args.workers}")
    print()

    def one(case: dict) -> dict:
        case_id = case["id"]
        category = case["category"]
        # 每条用自己的会话，避免互相污染——多轮题更要独立。
        #
        # thread 里必须带本次运行的批次号：checkpointer 的历史按 thread_id
        # 累积，固定线程名会让下一次评测接着上一次的对话继续聊（同一题
        # 被模型认为「已经问过 N 次」，回答变成精简版，关键词判分失败）。
        # 实测 eval-0044 在第 4 次重跑时 answers 出现「你已经连续问了三次」。
        thread_id = f"eval-{RUN_TAG}-{case_id}"

        started = time.perf_counter()
        answer = None
        error = None
        payload: dict = {}

        try:
            # 多轮题先重放前几轮，再拿最后一轮判分。
            for turn in case.get("turns") or []:
                post_ask(args.base_url, api_key, turn["content"], thread_id, args.timeout)

            payload = post_ask(args.base_url, api_key, case["query"], thread_id, args.timeout)
            answer = str(payload.get("answer", ""))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:160]
            error = f"HTTP {exc.code}: {detail}"
        except Exception as exc:
            # 截到 200 字：Milvus 的报错把真正的原因放在后半句
            # （「failed to create query plan」后面才是根因），
            # 截太短只剩前半句，看不出问题在哪。
            error = f"{type(exc).__name__}: {str(exc)[:200]}"

        latency = (time.perf_counter() - started) * 1000
        score = score_case(
            case,
            answer=answer,
            corpus_titles=titles,
            title_map=title_map,
            error=error,
        )

        # 意图识别和指代消解的准确率，从响应里顺带算——不用另造数据。
        expected_intent = CATEGORY_INTENT.get(category)
        intent_ok = None
        if expected_intent and payload.get("intent"):
            intent_ok = payload["intent"] == expected_intent

        item_ok = None
        if category == "coreference" and payload.get("resolved_item_id"):
            item_ok = payload["resolved_item_id"] in (case.get("expected_item_ids") or [])

        return {
            "id": case_id,
            "category": category,
            "passed": score.passed,
            "unjudged": score.unjudged,
            "checks": score.checks,
            "latency_ms": latency,
            "error": error,
            "mode": payload.get("mode"),
            "intent": payload.get("intent"),
            "expected_intent": expected_intent,
            "intent_ok": intent_ok,
            "resolved_item_id": payload.get("resolved_item_id"),
            "coref_ok": item_ok,
            "rewritten_query": payload.get("rewritten_query"),
            # 存全文，不截断。判分用的是完整回复，结果文件只留前几百字
            # 的话，事后想重新判分就失真了——实测有拒答措辞出现在
            # 400 字之后的，拿截断版重判会误判成失败。
            "answer": answer or "",
        }

    return run_batch(cases, one, args.workers)


# ---------------------------------------------------------------- 批跑与汇总


def run_batch(cases: list[dict], worker, workers: int) -> list[dict]:
    results: list[dict] = []
    done = 0
    total = len(cases)
    started = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, case): case for case in cases}
        for future in concurrent.futures.as_completed(futures):
            done += 1
            try:
                results.append(future.result())
            except Exception as exc:
                case = futures[future]
                results.append(
                    {
                        "id": case.get("id", "?"),
                        "category": case.get("category", "?"),
                        "passed": False,
                        "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    }
                )
            if done % 20 == 0 or done == total:
                elapsed = time.perf_counter() - started
                speed = done / elapsed if elapsed else 0
                eta = (total - done) / speed if speed else 0
                print(f"    {done}/{total}   剩约 {eta:.0f} 秒")

    return results


def rescore(args, cases: list[dict]) -> int:
    """对已有的结果文件重新判分。

    判据改了不必重跑。端到端一次 300 条要十几分钟外加大量模型调用，
    而结果文件里存了完整回复，够重新判分用。
    """

    path = Path(args.rescore)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_DIR / "data" / "eval" / args.rescore
    if not path.exists():
        print(f"结果文件不存在：{path}")
        return 1

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {case["id"]: case for case in cases}
    titles, title_map = load_corpus_titles()

    print("=" * 78)
    print(f"重新判分   {path.name}")
    print("=" * 78)
    print(f"记录    ：{len(rows)} 条")
    print(f"语料标题：{len(titles)} 个")
    print()

    rescored = []
    missing = 0
    for row in rows:
        case = by_id.get(row.get("id"))
        if case is None:
            missing += 1
            continue
        score = score_case(
            case,
            answer=row.get("answer"),
            corpus_titles=titles,
            title_map=title_map,
            error=row.get("error"),
        )
        rescored.append(
            {
                **row,
                "passed": score.passed,
                "unjudged": score.unjudged,
                "checks": score.checks,
            }
        )

    if missing:
        print(f"有 {missing} 条在测试集里找不到对应用例，已跳过。")
        print()

    print_summary(rescored, "rescore")
    print()
    print(f"结果已写入：{save_results(rescored, args, 'rescore')}")
    return 0


def save_results(results: list[dict], args, label: str) -> Path:
    """结果写盘。label 用来区分同一批用例的不同跑法。"""

    if args.out:
        base = Path(args.out)
        out_path = base.with_name(f"{base.stem}_{label}{base.suffix}")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = PROJECT_DIR / "data" / "eval" / f"result_{label}_{stamp}.jsonl"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        for item in results:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    return out_path


def print_summary(results: list[dict], mode: str) -> None:
    # 未判的排除在通过率之外。混进去会让报告虚高——
    # 检索模式下拒答题判不了「有没有硬凑文献」，算成通过就是白送 40 分。
    unjudged = [r for r in results if r.get("unjudged")]
    scored = [r for r in results if not r.get("skipped") and not r.get("unjudged")]

    if not scored:
        print("没有可判分的结果。")
        return

    print()
    print("=" * 78)
    print(f"总览（{mode} 模式）")
    print("=" * 78)

    passed = sum(1 for r in scored if r["passed"])
    latencies = [r["latency_ms"] for r in scored if r.get("latency_ms") is not None]
    print(f"  通过    {passed}/{len(scored)} = {passed / len(scored):.1%}")
    if latencies:
        print(
            f"  延迟    中位 {statistics.median(latencies):.0f} ms"
            f"    平均 {statistics.mean(latencies):.0f} ms"
            f"    最慢 {max(latencies):.0f} ms"
        )

    print()
    print(f"  {'类型':<18}{'通过':>12}{'通过率':>10}")
    print("  " + "-" * 44)
    by_category: dict[str, list[dict]] = {}
    for item in scored:
        by_category.setdefault(item["category"], []).append(item)

    for category in sorted(by_category, key=lambda c: -len(by_category[c])):
        group = by_category[category]
        ok = sum(1 for r in group if r["passed"])
        print(f"  {category:<18}{ok:>5}/{len(group):<6}{ok / len(group):>9.1%}")

    # 意图识别和指代消解，只在端到端模式下有。
    intent_judged = [r for r in scored if r.get("intent_ok") is not None]
    if intent_judged:
        ok = sum(1 for r in intent_judged if r["intent_ok"])
        print()
        print(f"  意图识别准确率   {ok}/{len(intent_judged)} = {ok / len(intent_judged):.1%}")
        wrong = [r for r in intent_judged if not r["intent_ok"]]
        for item in wrong[:5]:
            print(f"     {item['id']} 期望 {item['expected_intent']}，实际 {item['intent']}")

    coref_judged = [r for r in scored if r.get("coref_ok") is not None]
    if coref_judged:
        ok = sum(1 for r in coref_judged if r["coref_ok"])
        print(f"  指代消解准确率   {ok}/{len(coref_judged)} = {ok / len(coref_judged):.1%}")

    # 拒答单独提出来：这是幻觉的直接指标，藏在总通过率里看不见。
    refusals = by_category.get("refusal") or []
    if refusals:
        hallucinated = [
            r
            for r in refusals
            if not r["passed"] and (r.get("checks", {}).get("refusal", {}) or {}).get("cited_paper")
        ]
        print()
        print(f"  拒答失败（硬凑文献）：{len(hallucinated)}/{len(refusals)}")
        for item in hallucinated[:5]:
            cited = item["checks"]["refusal"]["cited_paper"]
            print(f"     {item['id']}  引了《{str(cited)[:24]}》")

    errors = [r for r in results if r.get("error")]
    if errors:
        print()
        print(f"  报错：{len(errors)} 条")
        for item in errors[:5]:
            print(f"     {item['id']}  {str(item['error'])[:90]}")

    if unjudged:
        print()
        print(f"  未判：{len(unjudged)} 条（不计入上面的通过率）")
        by_cat: dict[str, int] = {}
        for item in unjudged:
            by_cat[item["category"]] = by_cat.get(item["category"], 0) + 1
        for cat, num in sorted(by_cat.items()):
            print(f"     {cat:<18}{num:>4} 条")
        if mode == "retrieval":
            print("     检索模式拿不到最终回复。这类要 --mode e2e 才有结论。")

    print()
    print("=" * 78)
    print("失败明细")
    print("=" * 78)
    failures = [r for r in scored if not r["passed"]]
    if not failures:
        print("  全过。")
    for item in failures[: args_failure_limit]:
        checks = item.get("checks", {})
        reason = []
        if checks.get("retrieval") and not (
            checks["retrieval"].get("chunk_hit") or checks["retrieval"].get("item_hit")
        ):
            reason.append(f"没检索到（拿到 {checks['retrieval'].get('got_items', [])[:4]}）")
        if checks.get("keywords") and not checks["keywords"].get("skipped"):
            kw = checks["keywords"]
            reason.append(f"关键词 {kw['hit']}/{kw['total']}，缺 {kw.get('missing', [])[:3]}")
        if checks.get("refusal") and checks["refusal"].get("cited_paper"):
            reason.append(f"该拒答却引了《{str(checks['refusal']['cited_paper'])[:20]}》")
        if checks.get("metadata") and not checks["metadata"].get("passed"):
            reason.append(f"元数据只对上 {checks['metadata'].get('matched')}")
        if item.get("error"):
            reason.append(str(item["error"])[:70])
        print(f"  {item['id']:<12}{item['category']:<16}{'；'.join(reason) or '未达阈值'}")


args_failure_limit = 25
# 本次评测运行的批次号，拼进每个用例的 thread_id，保证跨运行不串话。
RUN_TAG = datetime.now().strftime("%m%d%H%M%S")


def main() -> int:
    global args_failure_limit

    parser = argparse.ArgumentParser(description="跑测试集，看系统表现")
    parser.add_argument(
        "--eval",
        default="eval_set.jsonl",
        help="测试集文件名或路径，默认 data/eval/eval_set.jsonl",
    )
    parser.add_argument("--mode", choices=["retrieval", "e2e"], default="retrieval")
    parser.add_argument("--category", default="", help="只跑某些类型，逗号分隔")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0 表示全跑")
    parser.add_argument("--workers", type=int, default=0, help="并发数，默认按模式定")
    parser.add_argument("--top-k", type=int, default=5, help="检索取前几条，默认 5")
    parser.add_argument(
        "--retrieval-mode",
        default="rrf",
        choices=["dense", "rrf", "compare"],
        help="检索方式。compare 会把两种都跑一遍并各出一份报告",
    )
    parser.add_argument("--collection", default="", help="集合名，默认取配置")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8030",
        help="API 地址，默认与 scripts/run_api.py 的监听端口一致",
    )
    parser.add_argument("--timeout", type=int, default=180, help="单条超时秒数")
    parser.add_argument("--out", default="", help="结果文件，默认按时间戳命名")
    parser.add_argument("--failures", type=int, default=25, help="打印多少条失败明细")
    parser.add_argument(
        "--rescore",
        default="",
        help="对已有结果文件重新判分，不重新请求。判据改了用这个，省一次重跑",
    )
    args = parser.parse_args()

    args_failure_limit = args.failures

    if not args.workers:
        args.workers = 4 if args.mode == "retrieval" else 3

    eval_path = Path(args.eval)
    if not eval_path.is_absolute() and not eval_path.exists():
        eval_path = PROJECT_DIR / "data" / "eval" / args.eval
    if not eval_path.exists():
        print(f"测试集不存在：{eval_path}")
        print("先生成：python scripts/build_eval_set.py")
        return 1

    categories = [c.strip() for c in args.category.split(",") if c.strip()]
    cases = load_cases(eval_path, categories or None, args.limit)

    print("=" * 78)
    print(f"跑测试集   {eval_path.name}   {args.mode} 模式")
    print("=" * 78)
    print(f"用例    ：{len(cases)} 条")
    if categories:
        print(f"类型    ：{', '.join(categories)}")
    print()

    if not cases:
        print("没有匹配的用例。")
        return 1

    if args.rescore:
        return rescore(args, cases)

    if args.mode == "retrieval":
        run_modes = (
            ["dense", "rrf"]
            if args.retrieval_mode == "compare"
            else [args.retrieval_mode]
        )
        for run_mode in run_modes:
            if len(run_modes) > 1:
                print()
                print("#" * 78)
                print(f"# 检索方式：{run_mode}")
                print("#" * 78)
            args.retrieval_mode = run_mode
            results = run_retrieval(cases, args)
            if not results:
                return 1
            print_summary(results, f"retrieval / {run_mode}")
            print()
            print(f"结果已写入：{save_results(results, args, f'retrieval_{run_mode}')}")

        if len(run_modes) > 1:
            print()
            print("=" * 78)
            print("两次跑的是同一批用例、同一套判分。往上翻，对比两张表。")
            print("  dense —— 纯中文向量检索")
            print("  rrf   —— 中文向量 + 英文 BM25，RRF 融合")
            print("=" * 78)
        return 0

    titles, title_map = load_corpus_titles()
    print(f"语料标题：{len(titles)} 个（拒答判定与跨语言判分用）")
    results = run_e2e(cases, args, titles, title_map)
    if not results:
        return 1

    print_summary(results, args.mode)
    print()
    print(f"结果已写入：{save_results(results, args, args.mode)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
