"""蒸馏扩充：调 DeepSeek 给每条种子生成意图保持的改写变体。

三道质检（对应 docs/project2-finetune-plan.md Week 1 规格）：
1. 规则过滤：长度界、规范化哈希去重（对种子表与批内都查重）；
2. 生成自检：prompt 内嵌——模型对每条变体自判意图，与种子意图不符即弃；
3. 人工抽检：脚本跑完后由人工分层抽样（Day 5，不在本脚本内）。

用法：
  python distill_variants.py --pilot                 # 每类抽 2 条试跑，写 distilled_pilot.jsonl
  python distill_variants.py                         # 全量，写 distilled.jsonl
  python distill_variants.py --workers 6             # 并发数（默认 4）

设计：
- 复用 app/llm.py 的 DeepSeek 配置（.env 在上级目录，config 会加载）；
- 温度 0.8（要多样性，与翻译那类确定性任务相反）；
- 每轮对每条种子一次调用生成 5 条变体；稀缺类多跑几轮（ROUNDS 表），
  不同轮侧重不同风格，压榨多样性；
- 结果逐条追加落盘，中断可续跑（已处理过的种子按 source_id 跳过）；
- 被拒变体写 dropped_log，可审计。
"""

import argparse
import json
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.json_utils import parse_llm_json  # noqa: E402
from app.llm import build_llm  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEEDS_PATH = DATA_DIR / "seeds.jsonl"

# 各类意图的训练轮数：种子越少轮数越多（每轮每条种子生成 5 条变体）。
# summary_ppt 只有 10 条种子，轮数有上限——变体多样性有天花板，
# 数量不够的部分宁可缺额也不灌水，Day 5 看分布再决定是否补模板合成。
ROUNDS = {
    "simple_qa": 2,
    "metadata_query": 4,
    "summary_paper": 4,
    "compare_papers": 4,
    "group_report": 5,
    "summary_ppt": 6,
}
VARIANTS_PER_CALL = 5
MAX_CHARS = 120
MIN_CHARS = 4

# 每轮的风格侧重，循环使用。
STYLE_NOTES = [
    "口语化、像真实用户随手打字，可以有语气词",
    "正式书面语，像研究报告里的问题",
    "省略式：省掉主语或宾语，短促但意图仍能判断",
    "换个疑问结构或句式（把疑问句改成陈述式求助、把「是什么」换成「介绍下」等）",
    "中英夹杂或使用学术表达",
    "换限定词与编号（换别的编号写法如「第X篇」「0XX号」，或换书名/页码等细节表述）",
]

PROMPT = """你是「论文资料问答系统」的数据增强助手。下面是一条真实用户问题及其意图标签。

请生成 {n} 条改写变体，要求：
1. 每条变体的意图必须仍然能判断为「{intent}」，不得改变含义；
2. 风格侧重：{style}；
3. 只使用原问题里已有的信息，不得引入新的文献编号、书名或主题；
4. 变体之间以及与原问题之间，措辞都要有差异。

输出 JSON（不要输出其他文字）：
{{"variants": [{{"text": "变体内容", "self_intent": "你判断这条变体属于的意图"}}, ...]}}

原问题：{question}
原意图：{intent}"""

NORMALIZE_RE = re.compile(r"[\s，。？！、；：""''「」（）,.?!;:'\"()]")


def normalize(text: str) -> str:
    return NORMALIZE_RE.sub("", (text or "").strip())


def qhash(text: str) -> str:
    import hashlib

    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()


def load_existing_hashes() -> set:
    """种子表的哈希集合：变体不得与任何种子字面重复。"""
    hashes = set()
    with open(SEEDS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                hashes.add(qhash(json.loads(line)["question"]))
    return hashes


def load_done_seed_keys(out_path: Path) -> set:
    """断点续跑：已处理过的 (source_id, round) 对。"""
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done.add((row["parent_id"], row["round"]))
    return done


class Distiller:
    def __init__(self, seed_hashes: set, done_keys: set, out_path: Path, drop_path: Path):
        settings = get_settings()
        # 温度 0.8：变体生成要多样性，和翻译那类确定性任务相反。
        self.llm = build_llm(settings).bind(temperature=0.8, max_tokens=900)
        self.seed_hashes = seed_hashes
        self.done_keys = done_keys
        self.out_path = out_path
        self.drop_path = drop_path
        self.lock = threading.Lock()
        self.counters = Counter()
        self.samples = []

    def generate(self, seed: dict, round_idx: int) -> None:
        key = (seed["source_id"], round_idx)
        if key in self.done_keys:
            return
        style = STYLE_NOTES[round_idx % len(STYLE_NOTES)]
        prompt = PROMPT.format(
            n=VARIANTS_PER_CALL, intent=seed["intent"], style=style,
            question=seed["question"],
        )
        try:
            resp = self.llm.invoke(prompt)
        except Exception as exc:
            with self.lock:
                self.counters[f"api_error:{seed['intent']}"] += 1
                self._drop(seed, round_idx, f"api_error: {exc}")
            return

        data = parse_llm_json(str(resp.content))
        if not isinstance(data, dict) or not isinstance(data.get("variants"), list):
            with self.lock:
                self.counters[f"bad_json:{seed['intent']}"] += 1
                self._drop(seed, round_idx, "模型输出不是合法 JSON")
            return

        kept_rows = []
        for v in data["variants"]:
            text = str(v.get("text", "")).strip().strip('"').strip("'")
            reason = None
            if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                reason = f"长度越界({len(text)})"
            elif v.get("self_intent") != seed["intent"]:
                reason = f"自检意图不符({v.get('self_intent')})"
            elif qhash(text) in self.seed_hashes:
                reason = "与种子重复"
            elif qhash(text) in {r["qhash"] for r in kept_rows}:
                reason = "批内重复"
            if reason:
                self._drop(seed, round_idx, reason, text)
                continue
            kept_rows.append(
                {"qhash": qhash(text), "text": text}
            )

        with self.lock:
            with open(self.out_path, "a", encoding="utf-8") as f:
                for row in kept_rows:
                    f.write(
                        json.dumps(
                            {
                                "question": row["text"],
                                "intent": seed["intent"],
                                "source": "distill",
                                "source_id": seed["source_id"],
                                "parent_question": seed["question"],
                                "style": style,
                                "round": round_idx,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                self.done_keys.add(key)
                self.counters[f"kept:{seed['intent']}"] += len(kept_rows)
                if len(self.samples) < 12:
                    self.samples.append(
                        (seed["intent"], seed["question"][:30], row_text := kept_rows[0]["text"] if kept_rows else "")
                    )

    def _drop(self, seed: dict, round_idx: int, reason: str, text: str = "") -> None:
        with self.lock:
            with open(self.drop_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"question": text, "intent": seed["intent"], "round": round_idx,
                         "reason": reason, "parent_id": seed["source_id"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="每类抽 2 条试跑")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    seeds = [json.loads(l) for l in open(SEEDS_PATH, encoding="utf-8") if l.strip()]
    if args.pilot:
        by_intent = {}
        for s in seeds:
            by_intent.setdefault(s["intent"], []).append(s)
        seeds = [s for group in by_intent.values() for s in random.Random(42).sample(group, 2)]
        out_path = DATA_DIR / "distilled_pilot.jsonl"
        print(f"[pilot] 抽样 {len(seeds)} 条种子，每条 1 轮")
        rounds = {intent: 1 for intent in by_intent}
    else:
        out_path = DATA_DIR / "distilled.jsonl"
        rounds = ROUNDS

    drop_path = out_path.with_name(out_path.stem + "_dropped.jsonl")
    out_path.touch(exist_ok=True)
    drop_path.touch(exist_ok=True)

    d = Distiller(load_existing_hashes(), load_done_seed_keys(out_path), out_path, drop_path)

    tasks = [
        (seed, r)
        for seed in seeds
        for r in range(rounds[seed["intent"]])
    ]
    print(f"任务数: {len(tasks)} 次调用（含断点跳过），并发 {args.workers}")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(d.generate, seed, r) for seed, r in tasks]
        done = 0
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 50 == 0:
                print(f"  进度 {done}/{len(tasks)}，{time.time() - t0:.0f}s")

    print(f"\n完成，用时 {time.time() - t0:.0f}s")
    kept = sum(v for k, v in d.counters.items() if k.startswith("kept"))
    print(f"保留变体: {kept}")
    for k, v in sorted(d.counters.items()):
        if not k.startswith("kept"):
            print(f"  拒绝 {k}: {v}")
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
