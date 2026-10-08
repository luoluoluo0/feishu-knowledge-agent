"""SFT 训练：Qwen3-1.7B LoRA 微调意图路由分类器（AutoDL 上运行）。

技术栈：Unsloth（省显存加速）+ TRL SFTTrainer + LoRA。

数据格式（docs/project2-finetune-plan.md）：
  system(任务说明+标签枚举) + user(query) → assistant(JSON 标签)
  本脚本直接读 splits 的 train/val.jsonl，用 tokenizer 的 chat template
  现场格式化（enable_thinking=False）——模板以机上实际权重为准，
  不在本地预拼，避免模板漂移。assistant 目标只放 {"intent": "..."}，
  短目标训练快、推理输出干净；reason 由集成层生成。

用法（首次跑默认配置）：
  python train_sft.py \
    --model-path /root/autodl-tmp/models/Qwen3-1.7B \
    --train-file /root/autodl-tmp/intent_route/data/train.jsonl \
    --val-file   /root/autodl-tmp/intent_route/data/val.jsonl \
    --output-dir /root/autodl-tmp/intent_route/outputs/sft_lr2e4_ep2_r16

调参扫描（Week 2 小扫描）改 --lr / --epochs / --lora-r 即可。
产出：output_dir 下合并后权重（merged_16bit）+ loss_curve.json。
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

SYSTEM_PROMPT = """你是论文资料问答系统的意图分类模块。判断用户问题属于哪一类意图，只输出 JSON。

可选意图：
1. simple_qa：普通问答，问某个事实、概念或结论。
2. metadata_query：元数据查询，问编号、整理者、DOI、附件、缺失情况、清单统计。
3. summary_paper：总结某篇论文，要研究问题、方法、数据、结论。
4. summary_ppt：总结某篇 PPT 或汇报内容。
5. group_report：生成组会汇报提纲。
6. compare_papers：对比多篇文献的异同。

输出格式：{"intent": "六选一"}"""


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/root/autodl-tmp/models/Qwen3-1.7B")
    parser.add_argument("--train-file", default="train.jsonl")
    parser.add_argument("--val-file", default="val.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--load-in-4bit", action="store_true", help="24G 卡默认全精度 LoRA；小卡训练再开")
    parser.add_argument("--no-unsloth", action="store_true", help="强制标准 PEFT/transformers 路线（不装 unsloth 也能跑）")
    args = parser.parse_args()

    from unsloth_import_fallback import load_model
    import torch
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    backend = "vanilla" if args.no_unsloth else "auto"
    model, tokenizer, peft_config = load_model(args, backend=backend)
    print(f"后端: {'unsloth' if peft_config is None else '标准 PEFT/transformers'}")

    print("[2/5] 用 chat template 格式化数据")
    def format_rows(rows: list[dict]) -> list[dict]:
        out = []
        for r in rows:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": json.dumps({"intent": r["intent"]}, ensure_ascii=False)},
            ]
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            out.append({"text": text})
        return out

    train_rows = load_jsonl(Path(args.train_file))
    val_rows = load_jsonl(Path(args.val_file))
    train_ds = Dataset.from_list(format_rows(train_rows))
    val_ds = Dataset.from_list(format_rows(val_rows))
    print(f"  train={len(train_ds)} val={len(val_ds)}")
    print(f"  train 意图分布: {dict(Counter(r['intent'] for r in train_rows))}")

    print("[3/5] 组装 Trainer")
    config_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        save_strategy="no",
        eval_strategy="epoch",
        per_device_eval_batch_size=args.batch_size,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        dataset_num_proc=4,
        packing=False,
    )
    if torch.cuda.is_bf16_supported():
        config_kwargs["bf16"] = True
    else:
        config_kwargs["fp16"] = True

    try:
        sft_config = SFTConfig(**config_kwargs)
        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer,
            train_dataset=train_ds, eval_dataset=val_ds, args=sft_config,
            peft_config=peft_config,
        )
    except TypeError as exc:
        # 旧版 trl：eval_strategy / SFTConfig 字段名不同时的兜底
        print(f"  SFTConfig 直建失败（{exc}），退回旧版参数组合")
        for key in ("eval_strategy", "evaluation_strategy", "dataset_text_field", "max_seq_length"):
            config_kwargs.pop(key, None)
        sft_config = SFTConfig(**config_kwargs)
        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer,
            train_dataset=train_ds, eval_dataset=val_ds, args=sft_config,
            dataset_text_field="text", max_seq_length=args.max_seq_len,
            peft_config=peft_config,
        )

    print("[4/5] 训练开始")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"训练完成，用时 {elapsed / 60:.1f} 分钟")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    loss_curve = [
        {"step": s.get("step"), "loss": s.get("loss"), "eval_loss": s.get("eval_loss"),
         "epoch": s.get("epoch"), "lr": s.get("learning_rate")}
        for s in trainer.state.log_history
    ]
    (out / "loss_curve.json").write_text(
        json.dumps({"args": vars(args), "elapsed_sec": elapsed, "log_history": loss_curve},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[5/5] 合并 LoRA 并保存")
    if peft_config is None:
        # unsloth 路线：专用合并接口
        model.save_pretrained_merged(str(out), tokenizer, save_method="merged_16bit")
    else:
        # 标准路线：merge_and_unload 后按普通 HF 模型保存
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(str(out), safe_serialization=True)
        tokenizer.save_pretrained(str(out))
    print(f"完成。合并权重在 {out}\n下一步: python eval_finetuned.py --model-path {out} --tag {Path(args.output_dir).name}")


if __name__ == "__main__":
    main()
