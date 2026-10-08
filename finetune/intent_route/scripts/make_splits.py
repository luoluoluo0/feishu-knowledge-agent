"""合并种子与蒸馏数据，做防泄漏切分，并生成人工抽检文件。

切分规则（docs/project2-finetune-plan.md Week 1 规格）：
- test 300：只从真实标注（eval_main/trap/prod 种子）分层抽取，冻结存档，
  绝不参与训练；脚本对已有 test.jsonl 拒绝覆盖（--force 才可）；
- val 150：剩余真实标注 + 蒸馏补足（真实为主）；
- train：其余全部蒸馏数据，大类按 CAP=1200 子采样（只采蒸馏行）；
- 防泄漏：规范化哈希全局去重，优先级 真实 > 蒸馏；切分前跨集查重。

产出：
- data/splits/{train,val,test}.jsonl
- data/spot_check.csv（每类 40 条分层抽检，供人工核对）
- outputs/splits_report.txt（分布报告）
"""

import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SPLITS_DIR = DATA_DIR / "splits"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

TEST_SIZE = 300
VAL_SIZE = 150
VAL_REAL_MIN = True  # val 以真实为主：真实全放（除进 test 的），蒸馏补足
TRAIN_CAP = 1200      # train 单类上限（只子采样蒸馏行）
SPOT_PER_CLASS = 40
SEED = 42

NORMALIZE_RE = re.compile(r"[\s，。？！、；：""''「」（）,.?!;:'\"()]")


def normalize(text: str) -> str:
    return NORMALIZE_RE.sub("", (text or "").strip())


def qhash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stratified_take(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """按类别比例分层抽 n 条（每类至少 1 条，若该类非空）。"""
    by_intent = defaultdict(list)
    for i, row in enumerate(pool):
        by_intent[row["intent"]].append(i)
    total = len(pool)
    alloc = {}
    for intent, idxs in by_intent.items():
        alloc[intent] = max(1, round(len(idxs) * n / total))
    # 调整到恰好 n：从超出比例最多的类削减
    while sum(alloc.values()) > n:
        intent = max(alloc, key=lambda k: alloc[k] / len(by_intent[k]))
        if alloc[intent] > 1:
            alloc[intent] -= 1
        else:
            break
    while sum(alloc.values()) < n:
        intent = min(alloc, key=lambda k: alloc[k] / len(by_intent[k]))
        alloc[intent] += 1

    picked = []
    for intent, k in sorted(alloc.items()):
        idxs = by_intent[intent]
        rng.shuffle(idxs)
        picked.extend(pool[i] for i in idxs[:k])
    return picked


def main() -> None:
    rng = random.Random(SEED)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    test_path = SPLITS_DIR / "test.jsonl"
    if test_path.exists() and "--force" not in sys.argv:
        sys.exit(f"test.jsonl 已存在（冻结），拒绝覆盖。确认要重切加 --force。")

    seeds = load_jsonl(DATA_DIR / "seeds.jsonl")
    distilled = load_jsonl(DATA_DIR / "distilled.jsonl")
    for row in seeds:
        row["_pool"] = "real"
    for row in distilled:
        row["_pool"] = "distill"

    # ---- 全局防泄漏去重：真实优先，蒸馏在后 ----
    seen: set[str] = set()
    real, dist = [], []
    for row in seeds + distilled:
        h = qhash(row["question"])
        if h in seen:
            continue
        seen.add(h)
        (real if row["_pool"] == "real" else dist).append(row)

    # ---- test：只从真实标注分层取 300，冻结 ----
    test = stratified_take(real, min(TEST_SIZE, len(real)), rng)
    test_ids = {id(r) for r in test}
    real_rest = [r for r in real if id(r) not in test_ids]

    # ---- val：剩余真实 + 蒸馏补足到 150 ----
    val = list(real_rest)
    if len(val) < VAL_SIZE:
        fill = stratified_take(dist, VAL_SIZE - len(val), rng)
        val_ids = {id(r) for r in fill}
        val.extend(fill)
    else:
        val = stratified_take(real_rest, VAL_SIZE, rng)
        val_ids = {id(r) for r in val}
    rest_ids = {id(r) for r in val}
    dist_rest = [r for r in dist if id(r) not in val_ids]

    # ---- train：蒸馏为主，大类子采样 ----
    train_by_intent = defaultdict(list)
    for r in dist_rest:
        train_by_intent[r["intent"]].append(r)
    train = []
    capped = {}
    for intent, rows in train_by_intent.items():
        rng.shuffle(rows)
        if len(rows) > TRAIN_CAP:
            rows = rows[:TRAIN_CAP]
            capped[intent] = TRAIN_CAP
        train.extend(rows)

    # ---- 落盘 ----
    def dump(path: Path, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                out = {k: v for k, v in r.items() if not k.startswith("_")}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    dump(test_path, test)
    dump(SPLITS_DIR / "val.jsonl", val)
    dump(SPLITS_DIR / "train.jsonl", train)

    # test 指纹：进 data_card，后续任何模型都在这份文件上评测
    test_bytes = test_path.read_bytes()
    test_sha = hashlib.sha256(test_bytes).hexdigest()[:16]

    # ---- 人工抽检文件：从三个集合合并池分层，每类 40 ----
    merged = test + val + train
    by_intent = defaultdict(list)
    for r in merged:
        by_intent[r["intent"]].append(r)
    spot = []
    for intent in sorted(by_intent):
        rows = by_intent[intent][:]
        rng.shuffle(rows)
        spot.extend(rows[:SPOT_PER_CLASS])
    rng.shuffle(spot)
    spot_path = DATA_DIR / "spot_check.csv"
    with open(spot_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "intent", "source", "你的判断(填意图)", "备注"])
        for r in spot:
            writer.writerow([r["question"], r["intent"], r["source"], "", ""])

    # ---- 报告 ----
    def dist_of(rows: list[dict]) -> Counter:
        return Counter(r["intent"] for r in rows)

    lines = [
        f"test={len(test)} val={len(val)} train={len(train)}",
        f"test sha256[:16] = {test_sha}（冻结，写进 data_card）",
        "",
        f"{'意图':<16}{'test':>5}{'val':>5}{'train':>7}{'合计':>7}",
    ]
    dt, dv, dtr = dist_of(test), dist_of(val), dist_of(train)
    for intent in sorted(set(dtr) | set(dv) | set(dt), key=lambda k: -(dt[k] + dv[k] + dtr[k])):
        lines.append(
            f"{intent:<16}{dt[intent]:>5}{dv[intent]:>5}{dtr[intent]:>7}{dt[intent]+dv[intent]+dtr[intent]:>7}"
        )
    if capped:
        lines.append(f"train 子采样到 {TRAIN_CAP} 的类: {capped}")
    lines.append(f"人工抽检: {len(spot)} 条 -> {spot_path.name}（每类 {SPOT_PER_CLASS}）")
    report = "\n".join(lines)
    print(report)
    (OUT_DIR / "splits_report.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
