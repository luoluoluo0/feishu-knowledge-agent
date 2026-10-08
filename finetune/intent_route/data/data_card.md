# 意图路由微调数据集 · 数据卡

生成日期：2026-09-27 ｜ 生成脚本：`scripts/extract_seeds.py` → `distill_variants.py` → `make_splits.py`

## 用途

微调 Qwen3-1.7B 做六分类意图路由（simple_qa / metadata_query / summary_paper / summary_ppt / group_report / compare_papers），替换现网 DeepSeek `classify_intent` 调用。

## 来源（三层金字塔）

| 层 | 来源 | 条数 | 性质 |
|---|---|---|---|
| 真实标注 | `data/eval/eval_set.jsonl` 经 CATEGORY_INTENT 映射 | 220 | 金标准（refusal/coreference/edge 无意图，排除） |
| 真实标注 | 陷阱集 intent_ok=True（generated_set_100 × result_gen） | 75 | 金标准；False 的 25 条在 `trap_review.csv` 待复核 |
| 真实标注 | `conversation_history.db` 生产查询 | 99 | 伪标签（去重后），抽检覆盖 |
| 蒸馏 | DeepSeek 对种子逐条生成变体（温度 0.8，6 风格轮换） | 5713 | 机器生成，生成即自检 + 规则过滤 |

种子合计 394，蒸馏 5713，全局规范化去重（SHA1，去空格标点）后合并 6107。

## 切分（脚本 make_splits.py，随机种子 42，可复现）

| 集合 | 条数 | 构成规则 |
|---|---|---|
| **test** | 300 | **只从真实标注分层抽取，冻结**。SHA256 前 16 位：`0395060a031876f7`——此后所有模型（DeepSeek 基线 / few-shot / 微调 / 量化）都在这同一份文件上评测 |
| val | 150 | 剩余真实（94）+ 蒸馏补足（56） |
| train | 4964 | 其余蒸馏；simple_qa 与 metadata_query 子采样至 1200 |

防泄漏：规范化哈希全局去重（真实优先）；test 先抽，val/train 从余量取。

## 各类分布

| 意图 | test | val | train | 合计 |
|---|---|---|---|---|
| simple_qa | 158 | 67 | 1200 | 1425 |
| metadata_query | 50 | 29 | 1200 | 1279 |
| summary_paper | 35 | 20 | 895 | 950 |
| compare_papers | 30 | 16 | 767 | 813 |
| group_report | 20 | 12 | 606 | 638 |
| summary_ppt | 7 | 6 | 296 | 309 |

## 质检记录

1. 规则过滤：长度界 4-120 字、标签枚举、规范化哈希去重——蒸馏阶段执行；
2. 生成自检：模型对每条变体自判意图，与种子意图不符即弃（通过率约 95%，拒绝明细 `distilled_dropped.jsonl`）；
3. 人工抽检：`spot_check.csv` 240 条（每类 40，三个集合混合），**待人工完成**——判定规则：任一类准确率 <95% → 该类重造。

## 已知局限

- **summary_ppt 只有 309 条**（10 条种子 × 6 轮 × 5 条的天花板），不到目标的 800。决策：接受缺额不灌水，macro-F1 口径下 300 条小类可学；若训练后该类 P/R 偏低，备选方案是模板合成（93 篇文献 × PPT 问句模板）；
- prod 来源是伪标签（模型自记的意图），靠抽检兜底；
- summary_ppt 的 test 覆盖只有 7 条，该类指标波动大，解读时注意。
