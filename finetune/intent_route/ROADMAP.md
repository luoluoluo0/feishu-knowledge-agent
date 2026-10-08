# 意图路由微调 · 全流程路线图

> 更新：2026-09-28。状态标记：✅ 完成 ｜ ▶ 当前要做 ｜ ⬜ 待做
> 目标：微调 Qwen3-1.7B 替换项目一的 DeepSeek 意图分类（95.67% / 1374ms），
> 简历产出「数据管线 + SFT + 量化部署 + 系统集成 + 三口径评测」完整闭环。

## 阶段 0：数据与基线 ✅

| 事项 | 状态 | 产出 |
|---|---|---|
| 种子提取（三源 394 条） | ✅ | data/seeds.jsonl |
| 蒸馏扩充（DeepSeek 生成 5713 条） | ✅ | data/distilled.jsonl |
| 防泄漏切分 | ✅ | splits/{train 4964, val 150, **test 300 冻结**}（sha 0395060a） |
| 基线一：DeepSeek 现网 | ✅ | **95.67% / macro-F1 95.01% / avg 1374ms**（要追平的尺子） |
| 数据卡 | ✅ | data/data_card.md |

## 阶段 1：基座基线（基线二）▶

目的：拿到「没微调能到多少」的锚点，证明提升幅度。面试必问。

```bash
# AutoDL（splits 与脚本已上传后）
python baseline_qwen_fewshot_hf.py --model-path /root/autodl-tmp/models/Qwen3-1.7B
```
产出：outputs/baseline_qwen_fewshot_hf_report.json → 填进对照表第 2 行。

## 阶段 2：训练与调参 ▶（本周）

LLaMA-Factory 路线（数据/YAML 已生成）：

```bash
pip install llamafactory
llamafactory-cli train intent_sft.yaml      # 首训：r16 / lr 2e-4 / 2ep，20-40 分钟
llamafactory-cli export intent_merge.yaml   # 合并 LoRA
python eval_finetuned.py --model-path /root/autodl-tmp/intent_route/outputs/lf_lr2e4_ep2_r16_merged --tag lf_lr2e4_ep2_r16
```

首训数字判断：
- macro-F1 ≥ 94：进小扫描——改 yaml 三行（learning_rate 1e-4/2e-4 × num_train_epochs 2/3 × lora_rank 16/32），每组合一个 output_dir + tag；
- macro-F1 < 90：先查数据与格式（看 eval_*.jsonl 的 raw 字段是不是输出格式跑偏），别急着扫参。

产出：**对照表**（DeepSeek / 基座 few-shot / 微调最优）+ 最优组合的混淆矩阵（与基线误判分布对照）。

## 阶段 3：量化与部署 ⬜

```bash
# 1. GGUF 转换两档（llama.cpp/convert_hf_to_gguf.py + 量化）
python convert_hf_to_gguf.py <merged_dir> --outfile intent-q4kM.gguf --outtype q8_0
llama-quantize intent-q8_0.gguf intent-q4_k_m.gguf Q4_K_M
# 2. 精度对照：两个 GGUF 各跑一遍 eval（把 eval_finetuned 换成走 llama-server 端点的小脚本，届时现写）
# 3. Ollama 打包：Modelfile（FROM intent-q4_k_m.gguf + SYSTEM 模板）→ ollama create
# 4. 压测：llama-server（Windows 原生）或 AutoDL vLLM —— 并发吞吐 / 首 token / 显存
```
产出：**部署报告四表**（量化精度损失 / 延迟 / 吞吐 / 显存）。

## 阶段 4：接入项目一 ⬜

- `app/intent.py` 的 `classify_intent` 加本地后端（配置开关选择 deepseek/local）；
- 本地失败自动降级回 DeepSeek（复用项目一的降级链路模式）；
- 跑项目一意图评测确认**不回退**；量测端到端延迟变化（预期单次问答少 1 次 LLM 调用、省 ~1.4s）。
产出：集成代码 + 端到端前后对照。

## 阶段 5：收尾 ⬜

- 可选加深：DPO 叠路由（偏好对来自混淆矩阵误判）；
- README（数据/训练/评测/部署四节 + 复现命令）；
- 简历 bullet 定稿（数字全部实测填入）+ interview-prep 加问答组。

## 待人工（随时，不阻塞）

- `data/spot_check.csv` 240 条人工抽检（<95% 整类重造）；
- `data/trap_review.csv` 25 条复核。

## 已定决策记录

- 训练在 AutoDL（4090），本地只做推理演示（Ollama, 1.2G 显存）；
- 不用 vLLM 起步（压测时可选 llama-server / 再议 vLLM）；
- 训练工具 LLaMA-Factory 为主（用户熟悉），自写 train_sft.py 留作对照/学习证据；
- summary_ppt 接受 309 条缺额不灌水；simple_qa/metadata train 超额已子采样至 1200；
- test 300 冻结（sha 0395060a），所有模型同尺评测。
