import argparse
import concurrent.futures
import json
import random
import re
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

try:
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.config import get_settings
    from app.eval_scoring import contains
    from app.llm import build_llm
except ModuleNotFoundError as exc:
    print(f"缺少依赖：{exc.name}")
    print("先 conda activate base 再跑。")
    raise SystemExit(1)


# 表格 / 图片 / 公式专项测试集。
#
# 为什么单独造这一套：
#
# 主测试集 300 条里，142/155 条期望命中纯文字块，只有 13 条落在表格块上，
# 而且那 13 条用正文就能答——实测新旧语料都是 13/13。也就是说主测试集
# 根本测不到新语料的卖点。
#
# 而换语料的全部理由就是这些东西：MinerU 从 PDF 里提出了 514 个表格、
# 216 张图片、655 个公式块，而旧语料（PyMuPDF）的三线表覆盖率只有
# 6.3%，图片是 0 条。
#
# 所以这套题的任务只有一条：**问那些只看了表格/图片/公式才能回答的问题**。
# 出题时强制模型说明「这条信息为什么只在表格里」，以此把靠正文就能答的
# 题挡在外面。
#
# 图片类的现状要提前知道：figure 块的 text 只有图注（二十来个字），
# 图片本体在 image_path 里，而 Agent 看不到图。所以图片类测的是
# 「Agent 对图片问题的能力边界」，预期会挂——挂了才是准确的。


CHUNKS_FILE = PROJECT_DIR / "data" / "processed" / "chunks_v2.jsonl"
DEFAULT_OUT = PROJECT_DIR / "data" / "eval" / "eval_special.jsonl"

# 每类出多少条。
PLAN = [
    ("table", 18),
    ("figure", 8),
    ("equation", 4),
]

# 一篇文献最多贡献几条，免得整卷被少数几篇占满。
MAX_PER_ITEM = 2

# 取样时多取一些当余量——出题有过滤，不是每个块都能变成题。
SAMPLING_HEADROOM = 10

TABLE_PROMPT = """下面是一篇学术论文里的一个表格。

请基于**表格里的具体数据**出一道检索测试题。

输出 JSON，四个字段：

- question：一句中文问题。必须问表格里的**具体数值或条目**
  （某个系数、样本量、时间区间、分组名称、编号等），
  让人非看这张表不可。绝对不要问「这篇文章讲了什么」这类
  看正文就能答的问题。
- reference_answer：一到三句中文答案，只依据表格内容。
- keywords：答案里的关键事实词，2 到 5 个，必须能在表格原文里原样找到。
  要挑**只有这张表里才有**的词，比如具体数值、专有名称。
- proof：一句话说明这条信息为什么**只能从表格里得到**，
  正文段落里没有。

只输出 JSON，不要解释。"""

EQUATION_PROMPT = """下面是一篇学术论文里的一个公式。

请基于这个公式出一道检索测试题。

输出 JSON，四个字段：

- question：一句中文问题，问这个公式的**含义或构成**
  （比如它定义了什么量、由哪些项组成、下标代表什么）。
  不要问「这篇文章讲了什么」这类泛泛的问题。
- reference_answer：一到三句中文答案，只依据公式本身和它的编号。
- keywords：答案里的关键事实词，2 到 5 个，必须能在原文里原样找到。
- proof：一句话说明为什么这条信息只能从这个公式得到。

只输出 JSON，不要解释。"""

FIGURE_PROMPT = """下面是一篇学术论文里的一张图的**图注**（图片本体看不到，
只有这行标题），以及它所在的章节。

请基于这张图出一道检索测试题。

输出 JSON，四个字段：

- question：一句中文问题，问这张图**展示了什么**。
  **必须带上文献标题或章节名作为限定**，写成「《论文标题》里的图 N 展示了什么」
  这样。原因：语料里多篇文献都有「图 1」「图 5」，只问「图 1 展示了什么」
  会有歧义，系统只能反问「你说的是哪一篇」——那是题目没出好，不是它答不出。
  实测第一批 8 条里挂了 5 条，全是这个原因：给了章节或地名限定的都答对了。
- reference_answer：一到三句中文答案。注意：你只能看到图注，
  所以答案只能依据图注的字面意思，不要推测图里的具体数据。
- keywords：答案里的关键事实词，2 到 5 个，必须能在原文里原样找到。
- proof：一句话说明这个问题为什么需要定位到这张图。

只输出 JSON，不要解释。"""


def load_chunks() -> list[dict]:
    return [
        json.loads(line)
        for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def call_json(llm, system: str, user: str, retries: int = 3) -> dict | None:
    """要模型吐 JSON。失败返回 None，不让单条失败拖垮整批。"""

    for attempt in range(retries):
        try:
            response = llm.invoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
            text = str(response.content).strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end < 0:
                raise ValueError("返回内容里没有 JSON")
            return json.loads(text[start : end + 1])
        except Exception as exc:
            if attempt == retries - 1:
                print(f"    生成失败：{type(exc).__name__} {str(exc)[:70]}")
                return None
            time.sleep(1.0 + attempt)
    return None


def keep_keyword(word: str) -> bool:
    text = str(word).strip()
    if not text:
        return False
    if re.search(r"[一-鿿]", text):
        return len(text) >= 2
    return len(text) >= 4


def build_case(llm, chunk: dict, category: str) -> dict | None:
    text = " ".join(str(chunk.get("text", "")).split())
    if category == "table" and len(text) < 150:
        return None
    if category == "figure" and len(text) < 8:
        return None
    if category == "equation" and len(text) < 20:
        return None

    if category == "table":
        system, payload = TABLE_PROMPT, text[:2500]
    elif category == "equation":
        system = EQUATION_PROMPT
        payload = f"公式（{chunk.get('section', '')}）：\n{text[:900]}"
    else:
        system = FIGURE_PROMPT
        payload = f"图注：{text}\n所在章节：{chunk.get('section', '')}\n论文标题：{chunk.get('title', '')}"

    data = call_json(llm, system, payload)
    if not data:
        return None

    question = str(data.get("question", "")).strip()
    answer = str(data.get("reference_answer", "")).strip()
    keywords = [
        k for k in (str(item).strip() for item in (data.get("keywords") or [])) if keep_keyword(k)
    ]
    proof = str(data.get("proof", "")).strip()

    if len(question) < 8 or not answer or len(keywords) < 2:
        return None

    # 关键事实词要能在原文里找到。找不到说明模型改写了，判分时会全部落空。
    missing = [k for k in keywords if not contains(text, k)]
    if len(missing) > len(keywords) / 2:
        return None

    return {
        "category": category,
        "language": "zh",
        "query": question,
        "expected_item_ids": [chunk.get("item_id", "")],
        "expected_chunk_ids": [chunk.get("chunk_id", "")],
        "reference_answer": answer,
        "answer_keywords": keywords,
        "keywords_missing": missing,
        # 出题时要求模型说明「为什么只能从这个块得到」。
        # 留着供人工抽查——它是不是真的只能从表格里答。
        "proof": proof,
        "source_excerpt": text[:300],
        "meta": {
            "block_type": chunk.get("block_type"),
            "section": chunk.get("section", ""),
            "page": chunk.get("page"),
            "title": chunk.get("title", ""),
            "image_path": chunk.get("image_path", ""),
        },
    }


def sample(chunks: list[dict], block_type: str, count: int) -> list[dict]:
    pool = [
        c
        for c in chunks
        if c.get("chunk_type") == "child" and c.get("block_type") == block_type
    ]
    random.shuffle(pool)

    per_item: dict[str, int] = {}
    picked = []
    for chunk in pool:
        item = str(chunk.get("item_id", ""))
        if per_item.get(item, 0) >= MAX_PER_ITEM:
            continue
        per_item[item] = per_item.get(item, 0) + 1
        picked.append(chunk)
        if len(picked) >= count:
            break
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description="生成表格/图片/公式专项测试集")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出文件")
    parser.add_argument("--workers", type=int, default=5, help="并发数")
    parser.add_argument("--seed", type=int, default=20260918, help="随机种子")
    parser.add_argument("--dry-run", action="store_true", help="只取样，不调模型")
    parser.add_argument(
        "--only",
        default="",
        help="只重新生成这类题（逗号分隔），其余从已有文件保留",
    )
    args = parser.parse_args()

    only = {item.strip() for item in args.only.split(",") if item.strip()}
    plan = [(name, count) for name, count in PLAN if not only or name in only]
    unknown = only - {name for name, _ in PLAN}
    if unknown:
        print(f"未知类别：{', '.join(sorted(unknown))}")
        print(f"可选：{', '.join(name for name, _ in PLAN)}")
        return 1

    random.seed(args.seed)

    chunks = load_chunks()
    if not chunks:
        print(f"语料为空：{CHUNKS_FILE}")
        return 1

    settings = get_settings()
    llm = None if args.dry_run else build_llm(settings)

    print("=" * 78)
    print("生成专项测试集")
    print("=" * 78)
    print(f"语料    ：{len(chunks)} 个块")
    print()

    cases: list[dict] = []

    for block_type, count in plan:
        picked = sample(chunks, block_type, count + SAMPLING_HEADROOM)
        print(f"  {block_type:<10}取样 {len(picked)} 块，出题中……")

        if args.dry_run:
            continue

        results: list[dict] = []
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(build_case, llm, chunk, block_type)
                for chunk in picked
            ]
            for future in concurrent.futures.as_completed(futures):
                done += 1
                try:
                    item = future.result()
                except Exception as exc:
                    print(f"    [{done}/{len(picked)}] 异常：{type(exc).__name__} {str(exc)[:60]}")
                    continue
                if item:
                    results.append(item)

        print(f"  {block_type:<10}出题 {len(results):>3} 条")
        cases.extend(results[:count])

    if args.dry_run:
        print()
        print("--dry-run：只取样，没有出题。")
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --only 时把其他类别从旧文件原样搬过来。这个测试集只有三十条，
    # 编号不稳定也无所谓，重排一遍就行。
    if only and out_path.exists():
        old = [
            json.loads(line)
            for line in out_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cases = [row for row in old if row.get("category") not in only] + cases

    for index, case in enumerate(cases, 1):
        case["id"] = f"sp-{index:04d}"

    with out_path.open("w", encoding="utf-8") as file:
        for case in cases:
            file.write(json.dumps(case, ensure_ascii=False) + "\n")

    import collections

    print()
    print("=" * 78)
    print("自检")
    print("=" * 78)
    by_cat = collections.Counter(c["category"] for c in cases)
    for name, _ in PLAN:
        print(f"  {name:<12}{by_cat.get(name, 0):>4}")
    print(f"  {'合计':<12}{len(cases):>4}")

    kw_bad = [c for c in cases if c.get("keywords_missing")]
    if kw_bad:
        print()
        print(f"  关键事实词部分找不到：{len(kw_bad)} 条（判分时会偏严）")

    print()
    print("样本示例：")
    for case in cases[:2]:
        print(f"    [{case['category']}] {case['query'][:70]}")
        print(f"      答案：{case['reference_answer'][:70]}")
        print(f"      只在表格里的理由：{case['proof'][:60]}")

    print()
    print(f"已写入：{out_path}")
    print()
    print("下一步：")
    print(f"  python scripts/run_eval.py --eval {out_path.name} --mode e2e")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
