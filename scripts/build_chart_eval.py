"""构建 30 条图表问答评测集（方案 A 环节 5）。

出题逻辑：从 figure_captions.jsonl 挑 high confidence 的图，按类型分层
（地图/关系图/流程图/表格截图/统计图），让 DeepSeek 按 key_claims 反向出题
——题的答案锚定在 caption 事实上，判分有依据。

三类题：
  图中事实（问某篇某图里的具体内容，答案在 key_claims 里）
  跨论文找图（描述图的content，问哪篇论文有这张图）
  图外陷阱（问图里没有的细节，期望答案承认不确定——幻觉率指标）

用法：python scripts/build_chart_eval.py [--count 30]
输出：data/eval/chart_qa_30.jsonl
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from app.config import get_settings  # noqa: E402
from app.llm import build_llm  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

CAPTIONS_PATH = PROJECT_DIR / "data" / "processed" / "figure_captions.jsonl"
OUT_PATH = PROJECT_DIR / "data" / "eval" / "chart_qa_30.jsonl"

# 各类型配额（总和 = count）。语料里 map 100 张是大头，按占比分层。
QUOTA = {"map": 10, "diagram": 8, "flowchart": 5, "table_image": 4, "stat": 3}
MAX_PER_PAPER = 2  # 单篇论文最多占几题，防同质化

FACT_PROMPT = """你是 RAG 评测出题助手。根据下面这张学术图表的解读信息，出 1 道「答案只看解读文字就能判分」的问答题。

图表信息：
- 论文编号：{item_id}，页码：{page}
- 原文图注：{paper_caption}
- 图表类型：{chart_type}
- 坐标轴：{axes}
- 图中事实（key_claims）：{claims}
- 概括：{summary}

要求：
1. 问题要像真实用户会问的（「{item_id} 篇的图里……」「……的地图上哪些地区……」），自然、具体
2. 答案必须能从上面的图中事实里找到——考图里的信息，不考论文其他部分
3. 给出 2~3 个判分关键词：正确答案的文本里应该出现的词（可以是答案里的关键实体/数值/地名）
4. 只输出 JSON：{{"question": "...", "expected_answer": "...", "keywords": ["...", "..."]}}"""

TRAP_PROMPT = """你是 RAG 评测出题助手。下面这张学术图表的解读信息里**没有**数值细节。请构造 1 道「看起来合理、但图里并没有明确答案」的陷阱题——考系统会不会硬编。

图表信息：
- 论文编号：{item_id}，页码：{page}
- 原文图注：{paper_caption}
- 图表类型：{chart_type}
- 图中事实（key_claims）：{claims}

要求：
1. 问一个具体数值/排名/比例类的细节（例如「最多的县是多少」「占比分别是多少」），但解读信息里没有这个数
2. 只输出 JSON：{{"question": "..."}}"""

FIND_PROMPT = """你是 RAG 评测出题助手。根据下面这张图的解读信息，出 1 道「跨论文找图」题：描述图的内容，问哪篇论文里有这张图。

图表信息：
- 论文编号：{item_id}
- 图表类型：{chart_type}
- 概括：{summary}
- 图中事实：{claims}

要求：
1. 问题里不要出现论文编号和论文标题（那是答案）
2. 描述要足够具体，让读过解读的人能对上是这张图
3. 只输出 JSON：{{"question": "...", "keywords": ["{item_id}"]}}——keywords 固定为论文编号"""


def call_json(llm, prompt: str) -> dict | None:
    """调 LLM 拿 JSON，失败重试一次。"""

    for _ in range(2):
        try:
            resp = llm.invoke([SystemMessage(content="只输出 JSON。"), HumanMessage(content=prompt)])
            text = str(resp.content).strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                return json.loads(text[start : end + 1])
        except Exception as exc:
            print(f"  出题失败重试：{exc}")
            time.sleep(2)
    return None


def pick_figures(count: int) -> list[dict]:
    """按类型分层挑图：只取 high confidence，单篇限流。"""

    rows = [json.loads(l) for l in CAPTIONS_PATH.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
    pool = [r for r in rows if (r.get("vlm") or {}).get("confidence") == "high"]
    by_type: dict[str, list[dict]] = {}
    for r in pool:
        t = r["vlm"].get("chart_type", "other")
        if t in ("bar_chart", "line_chart", "pie_chart"):
            t = "stat"
        by_type.setdefault(t, []).append(r)

    rng = random.Random(20260925)
    # 类型配额按比例缩放到 count
    total_quota = sum(QUOTA.values())
    quota = {t: max(1, round(count * q / total_quota)) for t, q in QUOTA.items()}
    while sum(quota.values()) > count:
        quota[max(quota, key=quota.get)] -= 1

    picked: list[dict] = []
    paper_used: Counter = Counter()
    for t, k in quota.items():
        candidates = by_type.get(t, [])
        rng.shuffle(candidates)
        got = 0
        for r in candidates:
            if got >= k:
                break
            iid = r.get("item_id")
            if paper_used[iid] >= MAX_PER_PAPER:
                continue
            picked.append(r)
            paper_used[iid] += 1
            got += 1
        print(f"  类型 {t}: 挑了 {got}/{k}（候选 {len(candidates)}）")
    rng.shuffle(picked)
    return picked[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()

    figures = pick_figures(args.count)
    if len(figures) < args.count:
        print(f"警告：只挑到 {len(figures)} 张（目标 {args.count}）")

    llm = build_llm(get_settings()).bind(temperature=0.3)

    cases = []
    for index, fig in enumerate(figures, start=1):
        vlm = fig["vlm"]
        ctx = {
            "item_id": fig.get("item_id"),
            "page": fig.get("page"),
            "paper_caption": str(fig.get("paper_caption") or "")[:200],
            "chart_type": vlm.get("chart_type"),
            "axes": json.dumps(vlm.get("axes"), ensure_ascii=False),
            "claims": json.dumps(vlm.get("key_claims"), ensure_ascii=False),
            "summary": vlm.get("summary"),
        }
        # 每 8 题配 1 道陷阱题（幻觉率指标）
        is_trap = index % 8 == 0
        prompt = TRAP_PROMPT.format(**ctx) if is_trap else (
            FIND_PROMPT.format(**ctx) if index % 11 == 0 else FACT_PROMPT.format(**ctx)
        )
        result = call_json(llm, prompt)
        if not result or not result.get("question"):
            print(f"  [{index}/{len(figures)}] 出题失败，跳过（{ctx['item_id']} p{ctx['page']}）")
            continue
        case = {
            "case_id": f"CQ-{index:03d}",
            "type": "trap" if is_trap else ("find_figure" if index % 11 == 0 else "figure_fact"),
            "item_id": ctx["item_id"],
            "image_path": fig.get("image_path"),
            "chart_type": ctx["chart_type"],
            "question": result["question"],
            "expected_answer": result.get("expected_answer", ""),
            "keywords": [str(k) for k in result.get("keywords", [])][:3],
            "trap": is_trap,
        }
        cases.append(case)
        print(f"  [{index}/{len(figures)}] {case['case_id']} {case['type']}: {case['question'][:40]}")

    OUT_PATH.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
    print(f"\n共 {len(cases)} 题 → {OUT_PATH.name}")
    print("类型分布:", dict(Counter(c["type"] for c in cases)))
    print("下一步：python scripts/run_chart_eval.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
