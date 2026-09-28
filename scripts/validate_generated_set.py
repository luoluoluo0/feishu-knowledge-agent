"""校验 generated_set_100.jsonl 是否满足生成提示词的全部自检规则与分布要求。"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

PATH = Path(__name__).parent / "data" / "eval" / "generated_set_100.jsonl"
INTENTS = {"simple_qa", "metadata_query", "summary_paper", "summary_ppt", "group_report", "compare_papers"}
TOOLS = {"hybrid_search_literature_card", "search_paper", "hybrid_search_ppt", "search_by_item",
         "search_all", "semantic_search_literature", "get_literature_by_item_id",
         "list_literature_by_reader", "list_missing_files", "list_literature_by_status"}
# 元数据类问题可用的等价工具族（专用工具比泛化卡片检索更精准，用哪个都对）
METADATA_TOOLS = {"hybrid_search_literature_card", "semantic_search_literature",
                  "get_literature_by_item_id", "list_literature_by_reader",
                  "list_missing_files", "list_literature_by_status"}
TRAPS = {
    "intent_list_vs_content_compare", "intent_summary_vs_groupreport", "intent_increment_followup",
    "intent_fallback_chitchat", "intent_metadata_wording", "planner_item_id_binding",
    "planner_multi_paper_compare", "planner_group_report_completeness", "planner_ppt_explicit",
    "planner_metadata_routing", "retrieval_paraphrase", "retrieval_keyword_exact",
    "retrieval_cross_lingual", "retrieval_scope_filter", "retrieval_summary_lowrank",
    "retrieval_no_answer", "coreference_rewrite", "clean_baseline",
}
NODES = {"coreference", "intent", "planner", "retrieval", "e2e"}
NUM_RE = re.compile(r"第\s*\d+\s*篇|#\d+|(?<![0-9#])\d{3}(?![0-9])")

cases = [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
errors = []
print(f"总条数: {len(cases)}")

ids = [c["case_id"] for c in cases]
if len(set(ids)) != len(ids):
    dup = [k for k, v in Counter(ids).items() if v > 1]
    errors.append(f"case_id 重复: {dup}")

for c in cases:
    cid, e = c["case_id"], c.get("expected", {})
    intent = e.get("intent")
    if intent not in INTENTS:
        errors.append(f"{cid}: intent 非法 {intent}")
    if e.get("intent_not") not in INTENTS:
        errors.append(f"{cid}: intent_not 非法 {e.get('intent_not')}")
    if e.get("intent_not") == intent:
        errors.append(f"{cid}: intent_not 与 intent 相同")
    if e.get("task_type") not in {"simple_qa", "summary_paper", "summary_ppt", "group_report", "compare_papers"}:
        errors.append(f"{cid}: task_type 非法 {e.get('task_type')}")
    tools = e.get("tools", [])
    if set(tools) - TOOLS:
        errors.append(f"{cid}: tools 含非法工具 {tools}")
    groups = e.get("tools_groups") or []
    for group in groups:
        if not set(group) <= TOOLS:
            errors.append(f"{cid}: tools_groups 含非法工具 {group}")
    # 闲聊/兜底用例可以一次检索都不触发（空 tools 合法），其余意图必须给出路径
    # 或等价工具组（tools_groups 优先作为判定依据）。
    if not tools and not groups and intent != "simple_qa":
        errors.append(f"{cid}: 非兜底意图 tools 为空")
    if intent == "metadata_query" and not (set(tools) & METADATA_TOOLS):
        errors.append(f"{cid}: metadata_query 未使用任何元数据类工具")
    if intent == "compare_papers":
        if not all(t == "search_by_item" for t in tools):
            errors.append(f"{cid}: compare_papers 工具应全为 search_by_item: {tools}")
        if len(e.get("item_ids", [])) < 2 or len(e.get("item_ids", [])) != len(tools):
            errors.append(f"{cid}: compare 的 item_ids 与步数不匹配 {e.get('item_ids')} vs {len(tools)}")
    if e.get("task_type") == "group_report" and not {"hybrid_search_literature_card", "search_paper", "hybrid_search_ppt"} <= set(tools):
        errors.append(f"{cid}: group_report 三工具不齐 {tools}")
    for iid in e.get("item_ids", []):
        if not (isinstance(iid, str) and re.fullmatch(r"\d{3}", iid)):
            errors.append(f"{cid}: item_ids 格式错误 {iid}")
    if NUM_RE.search(c["question"]) and not e.get("item_ids"):
        errors.append(f"{cid}: 问题点名编号但 item_ids 为空: {c['question']}")
    nodes = c.get("tested_nodes", [])
    if not set(nodes) <= NODES:
        errors.append(f"{cid}: tested_nodes 非法 {nodes}")
    if "retrieval" in nodes and not e.get("expected_refusal") and (
        not e.get("relevant_item_ids") or not e.get("evidence_keywords")
    ):
        # retrieval_no_answer 用例按规则 relevant 必须为空，豁免。
        errors.append(f"{cid}: retrieval 用例 relevant/keywords 为空")
    if e.get("expected_refusal") and e.get("relevant_item_ids"):
        errors.append(f"{cid}: refusal 用例 relevant 应为空")
    has_history = bool(c.get("history"))
    if has_history != ("rewrite" in e):
        errors.append(f"{cid}: history 与 rewrite 字段不匹配 (history={has_history})")
    for h in c.get("history", []):
        if not h.startswith("用户："):
            errors.append(f"{cid}: history 条目格式错误 {h}")
    if c.get("trap_type") not in TRAPS:
        errors.append(f"{cid}: trap_type 非法 {c.get('trap_type')}")
    cp = c.get("answer_checkpoints", [])
    if not (2 <= len(cp) <= 4):
        errors.append(f"{cid}: answer_checkpoints 数量 {len(cp)}")

intent_c = Counter(c["expected"]["intent"] for c in cases)
diff_c = Counter(c["difficulty"] for c in cases)
trap_c = Counter(c["trap_type"] for c in cases)
node_c = Counter(c["tested_nodes"][0] if len(c["tested_nodes"]) == 1 else "multi" for c in cases)
hist_n = sum(1 for c in cases if c.get("history"))
clean_n = trap_c.get("clean_baseline", 0)
trap_n = len(cases) - clean_n
pk = trap_c.get("retrieval_paraphrase", 0) + trap_c.get("retrieval_keyword_exact", 0)

print("意图分布:", dict(intent_c))
print("难度分布:", dict(diff_c))
print("节点主分布:", dict(node_c), "| 多节点:", sum(1 for c in cases if len(c['tested_nodes']) > 1))
print("陷阱覆盖:", len(trap_c), "类 | 陷阱用例:", trap_n, "| clean:", clean_n, "| history:", hist_n, "| paraphrase+keyword:", pk)
for t in sorted(TRAPS):
    if trap_c.get(t, 0) < 1:
        errors.append(f"trap_type 缺失: {t}")

if intent_c["simple_qa"] > 40:
    # 方案 A（2026-09-21）：编号+单点事实问题对齐系统设计改判 simple_qa
    # 后，simple_qa 占比约 35%，上限相应放宽。
    errors.append(f"simple_qa 超上限: {intent_c['simple_qa']}")
for it in INTENTS:
    if intent_c.get(it, 0) < 2:
        errors.append(f"意图 {it} 少于 2 条")
if pk < 4:
    errors.append("paraphrase+keyword 合计不足 4")
if not (10 <= clean_n <= 20):
    errors.append(f"clean_baseline 占比不合规: {clean_n}")
if trap_n < 60:
    errors.append(f"陷阱用例不足 60%: {trap_n}")
if not (15 <= hist_n <= 25):
    errors.append(f"history 用例数不在 15-25: {hist_n}")
for name, want, got in (("easy", 30, diff_c.get("easy", 0)), ("medium", 45, diff_c.get("medium", 0)), ("hard", 25, diff_c.get("hard", 0))):
    if abs(got - want) > 3:
        errors.append(f"难度 {name} 偏离目标超 3: 期望≈{want} 实际 {got}")

print()
if errors:
    print(f"发现 {len(errors)} 个问题:")
    for e in errors:
        print(" -", e)
    sys.exit(1)
print("✓ 全部自检规则与分布要求通过")
