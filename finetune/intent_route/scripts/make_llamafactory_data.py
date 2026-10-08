"""把意图路由数据集转成 LLaMA-Factory 的格式，并生成训练/合并 YAML。

复用上个项目（图表专家）验证过的模式：
- 数据：OpenAI messages 格式 jsonl（system/user/assistant），LF sharegpt
  格式化 + role/content 列映射（finetune/dataset_info.json 同款写法）；
- 注册：dataset_info.json 里登记 intent_route_train / intent_route_val；
- YAML：train（LoRA SFT）+ export（合并导出），路径按 AutoDL 约定写死
  /root/autodl-tmp/intent_route/...，与本目录其他脚本的 runbook 一致。

产出（全部写进 data/llamafactory/，yaml 在 outputs/llamafactory/）：
  intent_sft_train.jsonl / intent_sft_val.jsonl / dataset_info.json
  intent_sft.yaml    /  intent_merge.yaml

之后：
  scp 上传 → llamafactory-cli train intent_sft.yaml → llamafactory-cli export intent_merge.yaml
  → 用 eval_finetuned.py 在合并目录上出正式数字（与自写脚本路线同一把尺）。
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DATA_DIR = HERE / "data"
LF_DIR = DATA_DIR / "llamafactory"
OUT_DIR = HERE / "outputs" / "llamafactory"

# 与 train_sft.py 的 SYSTEM_PROMPT 保持一致——两条路线训出的模型
# 推理时看到的是同一个任务说明，评测才可比。
SYSTEM_PROMPT = """你是论文资料问答系统的意图分类模块。判断用户问题属于哪一类意图，只输出 JSON。

可选意图：
1. simple_qa：普通问答，问某个事实、概念或结论。
2. metadata_query：元数据查询，问编号、整理者、DOI、附件、缺失情况、清单统计。
3. summary_paper：总结某篇论文，要研究问题、方法、数据、结论。
4. summary_ppt：总结某篇 PPT 或汇报内容。
5. group_report：生成组会汇报提纲。
6. compare_papers：对比多篇文献的异同。

输出格式：{"intent": "六选一"}"""


def convert(split: str) -> str:
    rows = [json.loads(l) for l in open(DATA_DIR / "splits" / f"{split}.jsonl", encoding="utf-8") if l.strip()]
    out_name = f"intent_sft_{split}.jsonl"
    with open(LF_DIR / out_name, "w", encoding="utf-8") as f:
        for r in rows:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": json.dumps({"intent": r["intent"]}, ensure_ascii=False)},
            ]
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    print(f"{out_name}: {len(rows)} 条")
    return out_name


def main() -> None:
    LF_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_name = convert("train")
    val_name = convert("val")

    dataset_info = {
        "intent_route_train": {
            "file_name": train_name, "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"},
        },
        "intent_route_val": {
            "file_name": val_name, "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"},
        },
    }
    (LF_DIR / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train_yaml = f"""### 意图路由 LoRA SFT（LLaMA-Factory）
model_name_or_path: /root/autodl-tmp/models/Qwen3-1.7B
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.0
lora_target: all

dataset: intent_route_train
eval_dataset: intent_route_val
dataset_dir: /root/autodl-tmp/intent_route/data/llamafactory
template: qwen3
cutoff_len: 512
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 8

output_dir: /root/autodl-tmp/intent_route/outputs/lf_lr2e4_ep2_r16
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
save_strategy: no
eval_strategy: epoch
report_to: none
"""
    merge_yaml = f"""### 合并 LoRA 导出
model_name_or_path: /root/autodl-tmp/models/Qwen3-1.7B
adapter_name_or_path: /root/autodl-tmp/intent_route/outputs/lf_lr2e4_ep2_r16
template: qwen3
trust_remote_code: true
finetuning_type: lora
export_dir: /root/autodl-tmp/intent_route/outputs/lf_lr2e4_ep2_r16_merged
export_size: 5
export_device: cpu
export_legacy_format: false
"""
    (OUT_DIR / "intent_sft.yaml").write_text(train_yaml, encoding="utf-8")
    (OUT_DIR / "intent_merge.yaml").write_text(merge_yaml, encoding="utf-8")
    print(f"\nLLaMA-Factory 数据与配置已生成：\n  {LF_DIR}\n  {OUT_DIR}")
    print("上传后执行：\n  llamafactory-cli train intent_sft.yaml\n  llamafactory-cli export intent_merge.yaml\n"
          "  python eval_finetuned.py --model-path /root/autodl-tmp/intent_route/outputs/lf_lr2e4_ep2_r16_merged --tag lf_lr2e4_ep2_r16")


if __name__ == "__main__":
    main()
