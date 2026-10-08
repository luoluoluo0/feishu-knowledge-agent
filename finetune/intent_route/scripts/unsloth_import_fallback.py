"""模型加载双后端：unsloth 优先，未安装则自动退回标准 PEFT/transformers。

load_model(args, backend="auto"|"vanilla") 返回 (model, tokenizer, peft_config)：
- unsloth 路线：LoRA 已挂在模型上，peft_config 为 None，保存走 save_pretrained_merged；
- 标准路线：返回 LoraConfig，由 SFTTrainer(peft_config=...) 挂载，
  保存走 trainer.model.merge_and_unload()。

两条路线的 LoRA 超参完全一致（r / alpha=2r / 7 投影层 / dropout 0），
训练产物等价，unsloth 只影响速度与显存。
"""

import torch

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_model(args, backend: str = "auto"):
    if backend != "vanilla":
        try:
            from unsloth import FastLanguageModel

            print(f"[1/5] 加载模型（unsloth 后端）: {args.model_path}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model_path,
                max_seq_length=args.max_seq_len,
                dtype=None,
                load_in_4bit=args.load_in_4bit,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_r,
                lora_alpha=args.lora_r * 2,
                lora_dropout=0.0,
                target_modules=TARGET_MODULES,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            return model, tokenizer, None
        except ImportError:
            print("unsloth 未安装，退回标准 PEFT/transformers 路线")

    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[1/5] 加载模型（标准后端）: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.0,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return model, tokenizer, lora_config
