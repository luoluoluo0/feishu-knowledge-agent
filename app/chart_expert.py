"""图表专家：把微调模型挂成 Agent 的第 11 个工具（端到端接入）。

架构：DeepSeek 当总控，遇到图表细节问题（颜色/方位/数值/构成/标注）时
调用本工具。工具内部检索该图的 figure 块（增强后的【图表解读】文本），
按微调教材同款格式拼上下文，发给自部署的微调模型端点（v2 抗干扰教材
训练，拥挤上下文下鲁棒），返回图表事实给总控综合作答。

设计约束：
- 总闸 chart_expert_enabled 默认关闭：关闭时工具不注册，主路径零变化；
- 端点超时/异常一律兜底成文字说明，不抛异常——工具挂了 agent 还能
  基于检索到的【图表解读】自行作答，主链路永远可用；
- 提示词与 v2 教材逐字同构（SFT_SYSTEM + 图表资料 + 问题），训练分布
  = 部署分布；
- 返回文本尾部附 format_chunk 标准来源块——citations 解析器原生认得
  （资料N/标题/文献编号/图片行），前端缩略图与图命中指标都靠它；
- 同题缓存：陷阱题会诱发 agent 反复调用本工具（实测 17 次拖到 65s），
  同题直接命中缓存瞬时返回；资料不足时返回文本附「无需再检索」的
  明确表态，让总控尽早拒答。
"""

import json
import logging
import threading

import requests

from app.citations import current_citation_registry
from app.config import Settings
from app.tools import chunk_citation_key, format_chunk

logger = logging.getLogger(__name__)

# 与 finetune/make_sft_data.py 的 SFT_SYSTEM 逐字一致
SFT_SYSTEM = (
    "你是论文图表问答助手。只依据给定的图表资料回答问题；"
    "资料中没有的信息要如实说明，不要编造；"
    "图表资料与论文正文说法冲突时，以图表资料为准。"
)

# 与 run_chart_eval.py 的拒答词表一致：用于识别「资料中没有」类回答
REFUSAL_MARKS = ["没有", "未包含", "未提供", "未标注", "无法", "未给出", "未说明", "不包含"]

NO_FIGURE_HINT = (
    "图表专家：本次检索没有命中图表块，无法提供基于【图表解读】的作答。"
    "请改用其他检索工具获取资料后再回答。"
)
UNAVAILABLE_HINT = (
    "图表专家暂时不可用（{err}）。请基于检索到的【图表解读】自行作答。"
)

# 同题缓存：键=问题原文。命中即瞬时返回，专治陷阱题的工具调用螺旋。
# 工具调用在 langgraph 的线程池里并行执行，dict 的 iter+pop 序列在
# 并发下可能撞上「dictionary changed size during iteration」，加锁。
_CACHE_MAX = 128
_answer_cache: dict[str, str] = {}
_answer_cache_lock = threading.Lock()


def _cache_put(key: str, value: str) -> str:
    with _answer_cache_lock:
        if len(_answer_cache) >= _CACHE_MAX:
            _answer_cache.pop(next(iter(_answer_cache)))
        _answer_cache[key] = value
    return value


def ask_chart_expert(question: str, paper_tools, settings: Settings) -> str:
    """图表专家工具入口：检索 figure 块 → 调微调端点 → 返回图表事实。

    paper_tools 复用进程级检索栈（app/resources.py 单例），不新建连接。
    """

    cache_key = question.strip()
    cached = _answer_cache.get(cache_key)
    if cached is not None:
        logger.info("图表专家同题缓存命中：%s", question[:40])
        return cached

    # 1. 检索图表块。直连 store 绕过 rerank 门槛——图表问题问的是
    # 「图里的细节」，重排分天然偏低（相关性高≠分数高），走常规
    # _paper_retrieve 会被 0.8 门槛系统性误拒。top_k 取 8 再过滤，
    # 保证至少凑出 1~2 个 figure 块。
    try:
        result = paper_tools.paper_store.retrieve(
            question,
            top_k=8,
            item_id=None,
            expand_parents=False,
            min_score=0.0,
        )
    except Exception as exc:
        logger.warning("图表专家检索失败：%s", exc)
        return UNAVAILABLE_HINT.format(err=f"检索失败 {type(exc).__name__}")

    figure_chunks = [c for c in result.chunks if c.metadata.get("image_path")]
    if not figure_chunks:
        return _cache_put(cache_key, NO_FIGURE_HINT)

    # 2. 按教材同款格式拼上下文（最多 2 块，防上下文膨胀）。
    # figure 块的 text 本身就是「原图注 + 【图表解读】」增强文本。
    blocks = [f"图表资料：\n{c.text.strip()}" for c in figure_chunks[:2]]
    user = "\n\n".join(blocks) + f"\n\n问题：{question}"

    # 3. 调微调端点（OpenAI 兼容，本地隧道或同机服务）
    try:
        resp = requests.post(
            f"{settings.chart_expert_base_url.rstrip('/')}/chat/completions",
            json={
                "model": settings.chart_expert_model,
                "messages": [
                    {"role": "system", "content": SFT_SYSTEM},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
                "max_tokens": 512,
            },
            timeout=settings.chart_expert_timeout,
        )
        resp.raise_for_status()
        answer = str(resp.json()["choices"][0]["message"]["content"] or "").strip()
    except Exception as exc:
        # 端点故障是瞬时状态，不缓存，下次调用重试
        logger.warning("图表专家端点调用失败：%s", exc)
        return UNAVAILABLE_HINT.format(err=f"{type(exc).__name__}: {str(exc)[:60]}")

    if not answer:
        return _cache_put(cache_key, "图表专家没有返回有效内容，请基于检索到的【图表解读】自行作答。")

    # 4. 组装返回：结论在前（总控先读），来源块在后（引用面板与图命中
    #    指标从「资料N/图片：」行解析）。资料不足时附明确表态，让总控
    #    尽早拒答、不要继续检索。
    registry = current_citation_registry()
    source_blocks = []
    for i, chunk in enumerate(figure_chunks[:2], start=1):
        index = registry.number_for(chunk_citation_key(chunk)) if registry else i
        source_blocks.append(format_chunk(chunk, index))

    absent = any(m in answer for m in REFUSAL_MARKS)
    parts = [f"图表专家基于【图表解读】的回答（图文冲突时以它为准）：\n{answer}"]
    if absent:
        parts.append(
            "（图表解读已确认资料中未包含该信息：无需再检索其他工具，"
            "请直接如实告知用户图中未包含此信息。）"
        )
    parts.append("以下为专家所依据的图表资料原文（供引用标注与来源面板）：\n\n" + "\n\n".join(source_blocks))
    return _cache_put(cache_key, "\n\n".join(parts))
