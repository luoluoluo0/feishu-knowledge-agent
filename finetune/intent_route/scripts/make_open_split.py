"""开卷（open-set）评测的数据与配置生成器。

动机：冻结测试集 300 条里 95.7% 的父种子在训练集有蒸馏改写变体，
99.67% 是「闭卷分」。本脚本构造 seed 级开卷实验：

  1. 从冻结测试集分层抽 ~20% 条目 → open_eval（「未见模式」子集），
     其余 → open_seen（「见过模式」子集）；
  2. open_train = 主训练集剔除 held-out 种子的全部蒸馏变体；
  3. 生成 LLaMA-Factory 训练数据与 YAML（超参与主模型完全一致）。

之后训出的「开卷模型」分别在 open_seen 和 open_eval 上评测：
  open_seen − open_eval 的差值 = 改写泄漏（同源变体）带来的虚高；
  open_eval 本身 = 对全新问题模式的泛化能力（开卷分）。

冻结测试集 test.jsonl 原封不动——open_eval 只是它的派生子集
（报告里带 sha 溯源）。所有产出文件强制 LF 行尾（Windows 生成
CRLF 坑的根治，见 2026-09-30 排障记录）。

用法（本地跑）：
  python make_open_split.py
"""

import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

# SYSTEM_PROMPT 必须与主训练一字不差——开卷模型和主模型推理时
# 看到同一个任务说明，分数才可比。
from make_llamafactory_data import SYSTEM_PROMPT  # noqa: E402

DATA_DIR = HERE / "data"
SPLITS = DATA_DIR / "splits"
LF_DIR = DATA_DIR / "llamafactory"
OUT_DIR = HERE / "outputs" / "llamafactory"

HOLD_FRAC = 0.20   # 抽走 20% 模式：61 条开卷题（±13pp @95%CI 量级），训练集仍保留 80% 模式
SEED = 42

INTENTS = ["simple_qa", "metadata_query", "summary_paper", "summary_ppt", "group_report", "compare_papers"]


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump(path: Path, rows: list[dict]) -> None:
    # newline="\n"：强制 LF 行尾，根治 Windows 生成文件传 Linux 的 CRLF 坑
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            out = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def seed_key(row: dict) -> str:
    """种子的唯一标识。

    注意只能按 source_id 匹配：蒸馏行的 source 恒为 "distill"，
    父种子的来源信息没有跟过来，但 source_id 是原样继承的。
    """
    return row.get("source_id", "")


def main() -> None:
    rng = random.Random(SEED)
    SPLITS.mkdir(parents=True, exist_ok=True)
    LF_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    test = load(SPLITS / "test.jsonl")
    train = load(SPLITS / "train.jsonl")

    # ---- 1. 分层抽 held-out（按意图 ceil 比例，保证 summary_ppt 至少 2 条）----
    by_intent: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(test):
        by_intent[r["intent"]].append(i)
    held_idx: list[int] = []
    for intent in INTENTS:
        idxs = by_intent.get(intent, [])
        if not idxs:
            continue
        k = math.ceil(len(idxs) * HOLD_FRAC)
        rng.shuffle(idxs)
        held_idx.extend(idxs[:k])
    held_idx.sort()

    open_eval = [test[i] for i in held_idx]
    seen_set = set(held_idx)
    open_seen = [r for i, r in enumerate(test) if i not in seen_set]

    held_keys = {seed_key(r) for r in open_eval}

    # ---- 2. open_train：剔除 held-out 种子的全部蒸馏变体 ----
    removed = [r for r in train if seed_key(r) in held_keys]
    open_train = [r for r in train if seed_key(r) not in held_keys]
    # 防御断言：开卷训练集里绝不残留 held-out 变体
    assert not any(seed_key(r) in held_keys for r in open_train), "open_train 泄漏！"
    # 变体不得与任何真实问题字面相同（蒸馏阶段已保证，这里复检 held-out 部分）
    real_hashes = {hashlib.sha1(r["question"].strip().encode()).hexdigest() for r in open_eval}
    assert not any(
        hashlib.sha1(r["question"].strip().encode()).hexdigest() in real_hashes for r in removed
    ), "held-out 变体与开卷题字面重复！"

    # ---- 3. 落盘：splits 三个文件 + LF 训练数据 ----
    dump(SPLITS / "open_eval.jsonl", open_eval)
    dump(SPLITS / "open_seen.jsonl", open_seen)
    dump(SPLITS / "open_train.jsonl", open_train)

    with open(LF_DIR / "intent_sft_open_train.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in open_train:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": json.dumps({"intent": r["intent"]}, ensure_ascii=False)},
            ]
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    # dataset_info.json：在原有两个数据集基础上追加 open 注册（不覆盖别人的）
    info_path = LF_DIR / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    tags = {"role_tag": "role", "content_tag": "content",
            "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"}
    info["intent_route_train_open"] = {
        "file_name": "intent_sft_open_train.jsonl", "formatting": "sharegpt",
        "columns": {"messages": "messages"}, "tags": tags,
    }
    with open(info_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # ---- 4. 训练/合并 YAML：超参与主模型一致，仅换数据集与输出目录 ----
    train_yaml = f"""### 开卷模型 LoRA SFT（seed 级 held-out 20%）
model_name_or_path: /root/autodl-tmp/models/Qwen3-1.7B
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.0
lora_target: all

dataset: intent_route_train_open
eval_dataset: intent_route_val
dataset_dir: /root/autodl-tmp/intent_route/data/llamafactory
template: qwen3
cutoff_len: 512
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 8

output_dir: /root/autodl-tmp/intent_route/outputs/open_ep2_r16
per_device_train_batch_size: 16
per_device_eval_batch_size: 16
gradient_accumulation_steps: 1
learning_rate: 2.0e-4
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01
bf16: true
logging_steps: 10
save_strategy: "no"
eval_strategy: epoch
report_to: none

### 训练监控（不需要就删掉这 4 行）
use_swanlab: true
swanlab_project: llamafactory
swanlab_run_name: intent-open-ep2-r16
"""
    merge_yaml = """### 合并开卷模型 LoRA 导出
model_name_or_path: /root/autodl-tmp/models/Qwen3-1.7B
adapter_name_or_path: /root/autodl-tmp/intent_route/outputs/open_ep2_r16
template: qwen3
trust_remote_code: true
finetuning_type: lora
export_dir: /root/autodl-tmp/intent_route/outputs/open_ep2_r16_merged
export_size: 5
export_device: cpu
export_legacy_format: false
"""
    (OUT_DIR / "intent_sft_open.yaml").write_text(train_yaml, encoding="utf-8", newline="\n")
    (OUT_DIR / "intent_merge_open.yaml").write_text(merge_yaml, encoding="utf-8", newline="\n")

    # ---- 5. 报告 ----
    open_eval_sha = hashlib.sha256((SPLITS / "open_eval.jsonl").read_bytes()).hexdigest()[:16]
    lines = [
        f"held-out 比例: {HOLD_FRAC:.0%}（随机种子 {SEED}，可复现）",
        f"open_eval（未见模式）: {len(open_eval)} 条  | sha256[:16] = {open_eval_sha}",
        f"open_seen（见过模式）: {len(open_seen)} 条",
        f"open_train: {len(open_train)} 条（主训练集 {len(train)} − 剔除变体 {len(removed)}）",
        "",
        f"{'意图':<16}{'open_eval':>9}{'open_seen':>10}",
    ]
    ce = defaultdict(int)
    cs = defaultdict(int)
    for r in open_eval:
        ce[r["intent"]] += 1
    for r in open_seen:
        cs[r["intent"]] += 1
    for intent in INTENTS:
        lines.append(f"{intent:<16}{ce[intent]:>9}{cs[intent]:>10}")
    lines += [
        "",
        "判定口径：开卷模型在 open_seen 与 open_eval 各评一次，",
        "  open_seen − open_eval = 同源改写带来的虚高（泄漏量化）",
        "  open_eval 本身 = 对全新问题模式的泛化（开卷分）",
        "",
        "AutoDL 执行序列：",
        "  llamafactory-cli train intent_sft_open.yaml 2>&1 | tee train_open_ep2_r16.log",
        "  llamafactory-cli export intent_merge_open.yaml",
        "  python eval_finetuned.py --model-path /root/autodl-tmp/intent_route/outputs/open_ep2_r16_merged --test-file ../data/splits/open_seen.jsonl --tag open_seen",
        "  python eval_finetuned.py --model-path /root/autodl-tmp/intent_route/outputs/open_ep2_r16_merged --test-file ../data/splits/open_eval.jsonl --tag open_unseen",
    ]
    report = "\n".join(lines)
    print(report)
    (HERE / "outputs" / "open_split_report.txt").write_text(report, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
