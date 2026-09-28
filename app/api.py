import asyncio
import base64
import csv
import json
import logging
import queue
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.agent import (
    build_agent,
    build_langchain_tools,
    close_checkpointers,
    heal_dangling_tool_history,
    run_agent,
)
from app.auth import require_api_key
from app.checkpoint_store import clear_thread_checkpoints
from app.config import get_settings
from app.conversation_store import append_turn, clear_history
from app.coreference import Turn
from app.citations import (
    extract_all_tool_texts,
    extract_source_blocks_with_number,
    seed_registry_from_history,
)
from app.streaming import StreamEvent, iter_agent_events
from app.token_meter import read_token_usage


logger = logging.getLogger(__name__)
from app.errors import classify_exception
from app.logging_store import list_agent_logs, record_agent_log
from app.observability import (
    get_observability_status,
    initialize_observability,
    shutdown_observability,
)
from app.pipeline import (
    PLANNER_AGENT,
    TOOL_AGENT,
    prepare_question,
    route_for_intent,
)
from app.planner_agent import PlannerAgent, format_plan
from app.rate_limit import require_rate_limit
from app.resources import get_retrieval_stack


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动：初始化观测、调线程池、预热检索栈、跑数据清理；关闭：逐层释放。"""

    settings = get_settings()
    initialize_observability()

    # 同步端点跑在 anyio 线程池里，默认上限 40。每个 /ask 占一个线程
    # 8~30 秒，40 个并发就开始排队。调大只是缓解排队——真正的并发
    # 保护靠限流，彻底解决要任务队列（见 dev-notes 的取舍记录）。
    from anyio import to_thread

    to_thread.current_default_thread_limiter().total_tokens = (
        settings.app_threadpool_tokens
    )
    logger.info("线程池上限已设为 %d", settings.app_threadpool_tokens)

    from app.resources import close_retrieval_stack, get_retrieval_stack

    # 预热失败不能拖死启动：Milvus 没起时服务要照常可用（降级到
    # 首个请求时重试构建，再失败才在请求层报错——与预热上线前的
    # 行为一致）。实测教训：Milvus 容器挂着的时候服务整个起不来。
    try:
        get_retrieval_stack()
    except Exception:
        logger.warning("检索栈预热失败，将在首个请求时重试：", exc_info=True)

    # 数据保留清理：启动跑一次，之后守护线程每 24 小时一次。
    try:
        from app.retention import run_retention_cleanup, start_retention_daemon

        run_retention_cleanup(settings)
        start_retention_daemon(settings)
    except Exception:
        logger.warning("启动时数据保留清理失败：", exc_info=True)

    try:
        yield
    finally:
        shutdown_observability()
        close_checkpointers()
        close_retrieval_stack()


app = FastAPI(
    title="Feishu Paper Share Agent",
    version="0.1.0",
    lifespan=lifespan,
)
PROJECT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"

# 本地开发阶段先允许所有来源访问，方便 HTML 页面直接调用。
# 真正部署时，再把 allow_origins 改成你的前端域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

# 图表图片：来源面板缩略图与答案内嵌图走这里（MinerU 裁剪的 PNG）。
# 只挂 figures 目录——processed 下其余数据不对外。
FIGURES_DIR = PROJECT_DIR / "data" / "processed" / "figures"
if FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")


class ChatRequest(BaseModel):
    """前端或脚本发给后端的一次提问。"""

    question: str = Field(..., description="用户问题")
    thread_id: str = Field("feishu-web-user-1", description="会话 ID")
    include_trace: bool = Field(True, description="是否返回 Agent 调用工具的轨迹")


class ClearThreadRequest(BaseModel):
    """清空某个会话的多轮记忆。"""

    thread_id: str = Field(..., description="要清空的会话 ID")


@lru_cache
def get_tool_agent():
    """缓存 Tool Agent，让同一个服务进程可以继续多轮会话。"""

    return build_agent()


@lru_cache
def get_planner_agent():
    """缓存 PlannerAgent，避免每次请求都重新初始化模型和检索工具。"""

    return PlannerAgent()


def clear_cached_agent(mode: str) -> None:
    """清理缓存里的 Agent。

    Tool Agent 使用 checkpointer 保存多轮状态。
    如果工具调用过程中检索或模型接口失败，当前 thread_id 可能残留
    “AIMessage 有 tool_calls，但没有对应 ToolMessage”的半截历史。
    本地学习项目里最简单稳妥的处理方式，是失败后清掉缓存 Agent，
    让下一次请求重新创建干净的 Agent。

    注意这里只清 Agent，不动 checkpointer 的 SQLite 连接——连接是
    所有请求共享的，中途关闭会让在途请求集体报错（详见 app/agent.py）。
    """

    if mode == "tool_agent":
        get_tool_agent.cache_clear()
    elif mode == "planner_agent":
        get_planner_agent.cache_clear()


def simplify_message(message) -> dict[str, Any]:
    """把 LangChain Message 简化成前端和日志容易保存的字典。"""

    return {
        "type": getattr(message, "type", message.__class__.__name__),
        "name": getattr(message, "name", None),
        "content": getattr(message, "content", ""),
        "tool_calls": getattr(message, "tool_calls", []),
    }


def extract_sources_from_tool_text(tool_name: str, text: str) -> list[dict[str, Any]]:
    """从工具返回文本中提取前端可展示的来源信息。

    直接委托 citations 的解析——面板解析与 seed 历史续号同源，
    保证出处键一致。
    """

    return extract_source_blocks_with_number(tool_name, text)


def dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同一出处只留一条，保持首次出现顺序。"""

    unique_sources = []
    seen = set()
    for source in sources:
        key = (
            source.get("item_id", ""),
            source.get("title", ""),
            source.get("source_file", ""),
            source.get("location", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_sources.append(source)
    return unique_sources


def order_sources_by_citation(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """带引用编号的来源排前面、按编号升序，其余保持原序跟在后面。

    编号来自引用注册表，连续分配——所以编号前缀恰好就是
    [1]..[k]，前端答案里的徽标点过来一定有卡可对。
    """

    numbered = sorted(
        (source for source in sources if source.get("n") is not None),
        key=lambda source: source["n"],
    )
    unnumbered = [source for source in sources if source.get("n") is None]
    return numbered + unnumbered


def cap_sources_with_citations(
    sources: list[dict[str, Any]],
    answer: str,
    cap: int = 20,
) -> list[dict[str, Any]]:
    """截断来源清单，但答案里实际引用的编号必保留。

    长会话里编号按会话续接会一直涨，朴素的前 N 条截断会把本轮
    新来源或被引来源截掉，[n] 又变回悬空。策略：被引的必留，
    剩余名额按编号顺序补；被引数超过上限时引用优先（上限让位）。
    """

    ordered = order_sources_by_citation(dedupe_sources(sources))
    if len(ordered) <= cap:
        return ordered

    cited = {
        int(match)
        for match in re.findall(r"\[(?:资料)?(\d{1,2})\]", answer or "")
    }
    keep_cited = [source for source in ordered if source.get("n") in cited]
    rest = [
        source
        for source in ordered
        if source.get("n") not in cited
    ][: max(0, cap - len(keep_cited))]

    merged = sorted(
        keep_cited + rest,
        key=lambda source: (
            source.get("n") is None,
            source.get("n") if source.get("n") is not None else 0,
        ),
    )
    return merged


def build_trace_summary(messages, answer: str = "") -> dict[str, Any]:
    """整理 Agent 轨迹，给前端展示“调用了什么、依据是什么”。

    answer 用于截断时保住被引用的来源（cap_sources_with_citations）。
    """

    tool_calls = []
    sources = []

    for message in messages:
        for call in getattr(message, "tool_calls", []) or []:
            tool_calls.append(
                {
                    "name": call.get("name", ""),
                    "args": call.get("args", {}),
                }
            )

        if getattr(message, "type", "") == "tool":
            sources.extend(
                extract_sources_from_tool_text(
                    tool_name=getattr(message, "name", "") or "tool",
                    text=getattr(message, "content", "") or "",
                )
            )

    return {
        "tool_calls": tool_calls,
        "sources": cap_sources_with_citations(sources, answer),
    }


def simplify_step_result(step_result: dict) -> dict[str, Any]:
    """压缩 PlannerAgent 的每一步结果，避免接口和日志过长。"""

    result = step_result.get("result", "")
    return {
        "step_index": step_result.get("step_index"),
        "tool": step_result.get("tool"),
        "query": step_result.get("query"),
        "item_id": step_result.get("item_id"),
        "reason": step_result.get("reason"),
        "result_preview": result[:1500],
    }


def _safe_record_agent_log(**kwargs) -> Any:
    """写请求日志，失败只告警不外抛。

    日志库写不进去（磁盘满、SQLite 被锁）不该连累一次已经成功的回答，
    也不该掩盖真正要上报的业务异常。
    """

    try:
        log_id = record_agent_log(**kwargs)
    except Exception:
        logger.warning("请求日志写入失败（不影响本次请求）：", exc_info=True)
        return None

    # 长度监控的最后半步：token 已按请求入库（app/token_meter.py），
    # 这里补上超阈值告警——撞窗口/成本异常从被动发现变主动提示。
    usage = kwargs.get("token_usage") or {}
    total = usage.get("total_tokens") or 0
    threshold = get_settings().token_alert_threshold
    if total >= threshold:
        logger.warning(
            "单次请求 token 超阈值：thread=%s 合计 %d（输入 %d / 输出 %d，%d 次 LLM 调用），阈值 %d",
            kwargs.get("thread_id"),
            total,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            usage.get("llm_calls", 0),
            threshold,
        )
    return log_id


def log_failure_and_raise(
    *,
    mode: str,
    request: ChatRequest,
    exc: Exception,
    started_at: float,
) -> None:
    """记录失败日志，并把底层异常转成更清楚的接口错误。"""

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    error = classify_exception(exc)
    clear_cached_agent(mode)
    error_for_log = (
        f"[{error.error_type}] {error.message}\n"
        f"suggestion: {error.suggestion}\n"
        f"technical_detail: {error.technical_detail}"
    )

    _safe_record_agent_log(
        mode=mode,
        thread_id=request.thread_id,
        question=request.question,
        answer="",
        trace={},
        success=False,
        error=error_for_log,
        latency_ms=latency_ms,
    )

    raise HTTPException(
        status_code=error.status_code,
        detail=error.to_detail(),
    ) from exc


@app.get("/health")
def health():
    """健康检查：公开接口，只判断 FastAPI 服务是否启动。"""

    return {
        "status": "ok",
        "service": "feishu-paper-agent",
        "observability": get_observability_status(),
    }


@app.get("/")
def index():
    """返回前端聊天页面。"""

    index_file = FRONTEND_DIR / "agent_chat.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="前端页面不存在。")
    return FileResponse(index_file)


PROTECTED_DEPENDENCIES = [Depends(require_api_key), Depends(require_rate_limit)]


@app.get("/tools", dependencies=PROTECTED_DEPENDENCIES)
def list_tools():
    """查看当前 Tool Agent 可以调用哪些工具。"""

    tools = build_langchain_tools(paper_tools=get_retrieval_stack())
    return {
        "tools": [
            {
                "name": item.name,
                "description": item.description,
                "args_schema": item.args_schema.model_json_schema()
                if item.args_schema
                else None,
            }
            for item in tools
        ]
    }


@app.get("/admin/logs", dependencies=PROTECTED_DEPENDENCIES)
def admin_logs(limit: int = Query(20, ge=1, le=100)):
    """查看最近的 Agent 请求日志。"""

    return {"logs": list_agent_logs(limit=limit)}


@app.post("/admin/clear-thread", dependencies=PROTECTED_DEPENDENCIES)
def clear_thread(request: ClearThreadRequest):
    """清空某个 thread_id 的 Agent 多轮记忆和会话改写历史。"""

    clear_cached_agent("tool_agent")
    result = clear_thread_checkpoints(request.thread_id)
    result["deleted_turns"] = clear_history(request.thread_id)
    # db_path 是服务器文件系统的绝对路径，不该透给客户端。
    result.pop("db_path", None)
    return {
        "success": True,
        "message": "会话记忆已清空。",
        **result,
    }


@app.post("/admin/cleanup", dependencies=PROTECTED_DEPENDENCIES)
def admin_cleanup():
    """手动触发一轮数据保留清理，返回各项删除计数。"""

    from app.retention import run_retention_cleanup

    return {"success": True, "cleaned": run_retention_cleanup(get_settings())}


# --------------------------------------------------------------------------
# 管理总览：前端仪表盘要的服务状态、知识库规模、评测与入库摘要。
#
# 数据三处来，全部只读：
#   Milvus       —— 直接开轻量客户端查集合条数。不走检索栈单例：那玩意
#                   首次构建要 10 秒还会把 embedding 模型拉起来，总览页
#                   只是看一眼数字，不值得付这个代价；
#   本地数据文件 —— 入库 manifest、语料索引 CSV（路径与 ingest_paper.py
#                   里的定义保持一致）；
#   评测结果     —— data/eval 下按修改时间取最新一份 e2e / retrieval
#                   结果，逐行汇总出通过率、命中率和意图准确率。
# --------------------------------------------------------------------------

MANIFEST_FILE = PROJECT_DIR / "data" / "processed" / "ingest_manifest.jsonl"
CORPUS_INDEX_FILE = PROJECT_DIR / "data" / "metadata" / "literature_index_auto.csv"
EVAL_DIR = PROJECT_DIR / "data" / "eval"


def _milvus_stats() -> dict[str, Any]:
    """连一下 Milvus 查集合条数。容器没起时返回 ok=False 而不是 500。"""

    settings = get_settings()
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(
            uri=settings.milvus_uri, token=settings.milvus_token or None
        )
        try:
            if not client.has_collection(settings.milvus_collection, timeout=5):
                return {"ok": False, "error": f"集合 {settings.milvus_collection} 不存在"}
            stats = client.get_collection_stats(settings.milvus_collection, timeout=5)
            return {
                "ok": True,
                "collection": settings.milvus_collection,
                "chunks": int(stats.get("row_count", 0)),
            }
        finally:
            client.close()
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


def _parent_stats() -> dict[str, Any]:
    """父块条数来自本地 parents.json（懒加载，约 9MB，只在刷新时读一次）。"""

    try:
        from app.milvus_store import ParentStore

        return {"parents": ParentStore().load()}
    except Exception as exc:
        return {"parents": None, "error": str(exc)[:200]}


def _corpus_stats() -> dict[str, int]:
    """语料索引 CSV 的论文数（不含表头）。"""

    if not CORPUS_INDEX_FILE.exists():
        return {"papers": 0}
    with CORPUS_INDEX_FILE.open(encoding="utf-8-sig", newline="") as f:
        return {"papers": sum(1 for _ in csv.DictReader(f))}


def _manifest_stats() -> dict[str, Any]:
    """增量入库 manifest：条目数 + 最近一次入库记录。"""

    if not MANIFEST_FILE.exists():
        return {"entries": 0, "last": None}
    entries: dict[str, dict[str, Any]] = {}
    with MANIFEST_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("item_id"):
                entries[str(row["item_id"])] = row
    last = max(entries.values(), key=lambda e: str(e.get("ingested_at", ""))) if entries else None
    return {"entries": len(entries), "last": last}


def _latest_eval_result(prefix: str) -> Path | None:
    """按修改时间取最新一份评测结果（文件名里带时间戳，但排序只看 mtime）。"""

    candidates = sorted(
        EVAL_DIR.glob(f"result_{prefix}_*.jsonl"), key=lambda p: p.stat().st_mtime
    )
    return candidates[-1] if candidates else None


def _summarize_eval_file(path: Path) -> dict[str, Any]:
    """逐行汇总一份评测结果。e2e 和 retrieval 的字段取交集，有什么算什么。"""

    total = passed = unjudged = 0
    intent_ok = intent_total = 0
    ret_total = chunk_hits = item_hits = 0
    keyword_rates: list[float] = []
    latencies: list[float] = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            # 未判行（retrieval 模式下的 refusal）passed 与 unjudged 同时
            # 为真：分子不排除会把通过率算到 100% 以上（实测 262/260）。
            if row.get("passed") and not row.get("unjudged"):
                passed += 1
            if row.get("unjudged"):
                unjudged += 1
            latency = row.get("latency_ms")
            if isinstance(latency, (int, float)):
                latencies.append(float(latency))
            # edge 类没有期望意图，intent_ok 为 null，不计入分母。
            if row.get("intent_ok") is not None:
                intent_total += 1
                if row.get("intent_ok"):
                    intent_ok += 1
            checks = row.get("checks") or {}
            keywords = checks.get("keywords")
            if isinstance(keywords, dict) and isinstance(
                keywords.get("rate"), (int, float)
            ):
                keyword_rates.append(float(keywords["rate"]))
            retrieval = checks.get("retrieval")
            # 命中率只统计带期望文献的行：元数据/拒答/边界这些类别没有
            # 期望块，混进分母会把 92% 稀释成 61%（与报告页 by_category
            # 的过滤规则一致）。
            if isinstance(retrieval, dict) and retrieval.get("expected_items"):
                ret_total += 1
                if retrieval.get("chunk_hit"):
                    chunk_hits += 1
                if retrieval.get("item_hit"):
                    item_hits += 1

    judged = total - unjudged
    summary: dict[str, Any] = {
        "file": path.name,
        "finished_at": datetime.fromtimestamp(
            path.stat().st_mtime
        ).isoformat(timespec="seconds"),
        "total": total,
        "pass_rate": round(passed / judged, 4) if judged else None,
        "median_latency_ms": round(statistics.median(latencies)) if latencies else None,
    }
    if intent_total:
        summary["intent_accuracy"] = round(intent_ok / intent_total, 4)
    if keyword_rates:
        summary["keyword_rate"] = round(sum(keyword_rates) / len(keyword_rates), 4)
    if ret_total:
        summary["chunk_hit_rate"] = round(chunk_hits / ret_total, 4)
        summary["item_hit_rate"] = round(item_hits / ret_total, 4)
    return summary


def _eval_stats() -> dict[str, Any]:
    """测试集规模 + 最新一次 e2e / retrieval 结果摘要。"""

    result: dict[str, Any] = {
        "set_size": None,
        "special_size": None,
        "e2e": None,
        "retrieval": None,
    }
    for name, key in (("eval_set.jsonl", "set_size"), ("eval_special.jsonl", "special_size")):
        path = EVAL_DIR / name
        if path.exists():
            with path.open(encoding="utf-8") as f:
                result[key] = sum(1 for line in f if line.strip())
    for prefix, key in (("e2e", "e2e"), ("retrieval", "retrieval")):
        path = _latest_eval_result(prefix)
        if path is None:
            continue
        try:
            result[key] = _summarize_eval_file(path)
        except OSError:
            result[key] = None
    return result


@app.get("/admin/stats", dependencies=PROTECTED_DEPENDENCIES)
def admin_stats():
    """管理总览仪表盘：服务状态、知识库规模、评测与入库摘要（只读）。"""

    settings = get_settings()
    milvus = _milvus_stats()
    parents = _parent_stats()
    manifest = _manifest_stats()

    return {
        "services": {
            "backend": {"ok": True},
            "milvus": {**milvus, "parents": parents.get("parents")},
            "langfuse": get_observability_status(),
            "llm": {
                "configured": bool(settings.deepseek_api_key),
                "model": settings.deepseek_model,
            },
        },
        "knowledge": {
            "chunks": milvus.get("chunks") if milvus.get("ok") else None,
            "parents": parents.get("parents"),
            "papers": _corpus_stats().get("papers"),
            "manifest_entries": manifest.get("entries"),
        },
        "evals": _eval_stats(),
        "ingest": {"last": manifest.get("last")},
    }


# --------------------------------------------------------------------------
# RAG 评测报告：把 data/eval 里的历史评测结果聚合成「类别 × 方案」的
# 对比视图。全部只读——检索方案对比用的是 calibration 时期留下的真实
# 运行结果（dense / rrf / weighted），不现跑。
#
# 指标口径（与 scripts/run_eval.py 的判分一致）：
#   recall   = item_hit 率（期望文献进了 top-k）
#   mrr      = 1/hit_rank 的均值（期望文献排得越靠前越高）
# 只有带期望文献的类别（en_fact / zh_fact / cross_lingual）参与检索指标；
# 多轮题在检索模式下是 skipped 占位行，天然被过滤。
# --------------------------------------------------------------------------

_VARIANT_LABELS = {
    "dense": "纯向量",
    "rrf": "RRF 混合",
    "weighted": "加权融合",
    "baseline": "基线",
    "regression": "回归复跑（当前语料）",
}


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _retrieval_file_stats(path: Path) -> dict[str, Any]:
    """按类别汇总一份检索结果。

    两个指标的口径故意不同（与 run_eval 的双层判分一致）：
    - recall：item_hit 率——期望**文献**进了 top-k，宽口径；
    - mrr：期望**块**的排名倒数均值（hit_rank 只在 chunk_hit 时有值），
      严口径。item_hit 为真时期望块可能没命中，所以 MRR 必须按
      chunk_hit 算，用 item_hit 会得到 0。
    """

    cats: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            retrieval = (row.get("checks") or {}).get("retrieval")
            if not isinstance(retrieval, dict):
                continue  # 多轮题的 skipped 占位行没有 checks
            if not retrieval.get("expected_items"):
                continue  # 没有期望文献的类别（元数据/拒答/总结等）不参与
            cat = str(row.get("category") or "?")
            s = cats.setdefault(cat, {"n": 0, "hits": 0, "chunk_hits": 0, "rr": 0.0})
            s["n"] += 1
            rank = retrieval.get("hit_rank")
            if retrieval.get("item_hit"):
                s["hits"] += 1
            if retrieval.get("chunk_hit"):
                s["chunk_hits"] += 1
                if isinstance(rank, (int, float)) and rank > 0:
                    s["rr"] += 1.0 / rank
    by_category = {
        cat: {
            "n": int(s["n"]),
            "recall": round(s["hits"] / s["n"], 4) if s["n"] else None,
            "mrr": round(s["rr"] / s["n"], 4) if s["n"] else None,
        }
        for cat, s in sorted(cats.items())
    }
    n_all = sum(s["n"] for s in by_category.values())
    return {
        "by_category": by_category,
        "overall": {
            "n": n_all,
            "recall": round(sum(s["hits"] for s in cats.values()) / n_all, 4) if n_all else None,
            "mrr": round(sum(s["rr"] for s in cats.values()) / n_all, 4) if n_all else None,
        },
    }


def _retrieval_variants() -> list[dict[str, Any]]:
    """每个检索方案取 mtime 最新的一份结果，聚合成对比数据。"""

    best: dict[str, Path] = {}
    for path in EVAL_DIR.glob("result_retrieval_*.jsonl"):
        stem = path.stem[len("result_retrieval_"):]
        # 文件名两种形制：result_retrieval_<方案>_<时间戳> 与
        # result_retrieval_<纯时间戳>（最早的基线跑）。先把结尾时间戳
        # 摘掉，剩下空的就是基线——剩下的若还是纯时间戳，说明整个
        # stem 就是个时间戳，同样是基线。
        variant = re.sub(r"_\d{8}_\d{6}$", "", stem)
        if re.fullmatch(r"\d{8}_\d{6}", variant):
            variant = "baseline"
        if not variant:
            variant = "baseline"
        # 语料变更后的回归复跑不是新检索方案：单独归成一档，避免被当成
        # 第五个方案与 RRF 并列（2026-09-22 回归文件名里还带着内层
        # 时间戳与 _retrieval_rrf 后缀，掉不进上面任何规则）。
        if variant not in _VARIANT_LABELS and "regression" in variant:
            variant = "regression"
        if variant not in best or path.stat().st_mtime > best[variant].stat().st_mtime:
            best[variant] = path
    ordered = sorted(best.items(), key=lambda kv: kv[1].stat().st_mtime)
    variants = []
    for variant, path in ordered:
        stats = _retrieval_file_stats(path)
        variants.append(
            {
                "key": variant,
                "label": _VARIANT_LABELS.get(variant, variant),
                "file": path.name,
                "finished_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                **stats,
            }
        )
    return variants


def _e2e_file_stats(path: Path) -> dict[str, Any]:
    """按类别汇总一份 e2e 结果：通过率与意图准确率。"""

    cats: dict[str, dict[str, float]] = {}
    latencies: list[float] = []
    total = passed = unjudged = 0
    intent_ok = intent_total = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            # 未判行（retrieval 模式下的 refusal）passed 与 unjudged 同时
            # 为真：分子不排除会把通过率算到 100% 以上（实测 262/260）。
            if row.get("passed") and not row.get("unjudged"):
                passed += 1
            if row.get("unjudged"):
                unjudged += 1
            latency = row.get("latency_ms")
            if isinstance(latency, (int, float)):
                latencies.append(float(latency))
            cat = str(row.get("category") or "?")
            s = cats.setdefault(cat, {"n": 0, "passed": 0, "intent_ok": 0, "intent_total": 0})
            s["n"] += 1
            if row.get("passed"):
                s["passed"] += 1
            if row.get("intent_ok") is not None:
                s["intent_total"] += 1
                if row.get("intent_ok"):
                    s["intent_ok"] += 1
                    intent_ok += 1
                intent_total += 1
    judged = total - unjudged
    by_category = {
        cat: {
            "n": int(s["n"]),
            "pass_rate": round(s["passed"] / s["n"], 4) if s["n"] else None,
            "intent_accuracy": round(s["intent_ok"] / s["intent_total"], 4) if s["intent_total"] else None,
        }
        for cat, s in sorted(cats.items())
    }
    return {
        "file": path.name,
        "finished_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "total": total,
        "pass_rate": round(passed / judged, 4) if judged else None,
        "intent_accuracy": round(intent_ok / intent_total, 4) if intent_total else None,
        "median_latency_ms": round(statistics.median(latencies)) if latencies else None,
        "by_category": by_category,
    }


@app.get("/admin/eval-report", dependencies=PROTECTED_DEPENDENCIES)
def admin_eval_report():
    """评测报告页：KPI 元信息 + 检索方案对比 + e2e 类别指标 + LLM 评判。"""

    settings = get_settings()
    e2e_path = _latest_eval_result("e2e")
    from app.llm_judge import latest_judged

    return {
        "meta": {
            "set_size": _count_jsonl(EVAL_DIR / "eval_set.jsonl"),
            "special_size": _count_jsonl(EVAL_DIR / "eval_special.jsonl"),
            "papers": _corpus_stats().get("papers"),
            "embeddings_model": settings.embeddings_model,
            "rerank_model": settings.rerank_model if settings.rerank_enabled else None,
            "model": settings.deepseek_model,
        },
        "retrieval": {"variants": _retrieval_variants()},
        "e2e": _e2e_file_stats(e2e_path) if e2e_path else None,
        "judge": latest_judged(),
    }


# --------------------------------------------------------------------------
# 观测页：意图成本（Langfuse Metrics API）+ 评估趋势（本地结果文件）
# + 阈值扫描（scripts/run_threshold_sweep.py 预生成的曲线数据）。
#
# Langfuse 是 v4 events_only 模式：旧 /traces、/observations 查询接口
# 整个被禁用，token 聚合走新的 GET /api/public/v2/metrics（服务端聚合，
# 按 tags 分组）。意图维度依赖请求时打到 trace 上的 intent:* 标签
# （见 observability.attach_intent_tag），标签落地前的数据归「未标注」。
# --------------------------------------------------------------------------


def _intent_cost(settings) -> dict[str, Any]:
    """调 Langfuse Metrics API v2，按 intent 标签聚合近 7 天 token。"""

    if not settings.langfuse_tracing_enabled:
        return {"error": "Langfuse 未启用（LANGFUSE_TRACING_ENABLED=false）"}

    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    now = datetime.now()
    query = {
        "view": "observations",
        "metrics": [
            {"measure": "totalTokens", "aggregation": "sum"},
            {"measure": "count", "aggregation": "count"},
        ],
        "dimensions": [{"field": "tags"}],
        "filters": [],
        "fromTimestamp": (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "toTimestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {"row_limit": 200},
    }
    url = settings.langfuse_base_url.rstrip("/") + "/api/public/v2/metrics"
    token = base64.b64encode(
        f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
    ).decode()
    request = Request(
        f"{url}?{urlencode({'query': json.dumps(query)})}",
        headers={"Authorization": f"Basic {token}"},
    )
    try:
        with urlopen(request, timeout=8) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as exc:
        return {"error": f"Langfuse 查询失败：{str(exc)[:160]}"}

    by_intent: dict[str, dict[str, int]] = {}
    for row in payload.get("data", []):
        tags = row.get("tags") or []
        intent = next(
            (t[len("intent:"):] for t in tags if isinstance(t, str) and t.startswith("intent:")),
            None,
        )
        bucket = by_intent.setdefault(intent or "未标注", {"tokens": 0, "llm_calls": 0})
        bucket["tokens"] += int(row.get("sum_totalTokens") or 0)
        bucket["llm_calls"] += int(row.get("count_count") or 0)
    total_tokens = sum(b["tokens"] for b in by_intent.values())
    total_calls = sum(b["llm_calls"] for b in by_intent.values())
    ranked = sorted(by_intent.items(), key=lambda kv: -kv[1]["tokens"])
    for _, bucket in ranked:
        bucket["share"] = round(bucket["tokens"] / total_tokens, 4) if total_tokens else None

    top = ranked[0] if ranked else None
    return {
        "window_days": 7,
        "fetched_at": now.isoformat(timespec="seconds"),
        "total_tokens": total_tokens,
        "total_llm_calls": total_calls,
        "top_intent": {
            "intent": top[0],
            "tokens": top[1]["tokens"],
            "share": top[1].get("share"),
        } if top and top[0] != "未标注" else None,
        "by_intent": [{"intent": name, **bucket} for name, bucket in ranked],
    }


def _eval_trend() -> list[dict[str, Any]]:
    """历史评测结果按文件时间连成趋势点：e2e 通过率 / 检索 recall。"""

    points: list[dict[str, Any]] = []
    for path in sorted(EVAL_DIR.glob("result_*.jsonl"), key=lambda p: p.stat().st_mtime):
        stem = path.stem
        if stem.startswith("result_retrieval_"):
            kind = "retrieval"
        elif stem.startswith(("result_e2e_", "result_rescore_")):
            kind = "e2e"
        else:
            continue
        try:
            rows = [
                json.loads(line)
                for line in path.open(encoding="utf-8")
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError):
            continue

        if kind == "retrieval":
            n = hits = 0
            for row in rows:
                retrieval = (row.get("checks") or {}).get("retrieval")
                if isinstance(retrieval, dict) and retrieval.get("expected_items"):
                    n += 1
                    if retrieval.get("item_hit"):
                        hits += 1
            if n:
                points.append(
                    {
                        "file": path.name,
                        "ts": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                        "kind": kind,
                        "n": n,
                        "recall": round(hits / n, 4),
                    }
                )
        else:
            judged = [row for row in rows if not row.get("unjudged")]
            if judged:
                passed = sum(1 for row in judged if row.get("passed"))
                points.append(
                    {
                        "file": path.name,
                        "ts": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                        "kind": kind,
                        "n": len(judged),
                        "pass_rate": round(passed / len(judged), 4),
                    }
                )
    return points


def _threshold_sweep() -> dict[str, Any] | None:
    path = EVAL_DIR / "threshold_sweep.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@app.get("/admin/observe", dependencies=PROTECTED_DEPENDENCIES)
def admin_observe():
    """观测页：意图成本 + 评估趋势 + 阈值扫描曲线。"""

    settings = get_settings()
    return {
        "intent_cost": _intent_cost(settings),
        "eval_trend": _eval_trend(),
        "threshold_sweep": _threshold_sweep(),
    }


# --------------------------------------------------------------------------
# LLM 评判任务：POST 后台线程跑（95 条约 1 分钟），GET 轮询进度。
# 结构照抄论文入库的任务模式：全局同时只允许一个任务。
# --------------------------------------------------------------------------

class JudgeRequest(BaseModel):
    limit: int | None = Field(None, ge=1, le=500, description="只评判前 N 条，调试用")


JUDGE_TASKS: dict[str, dict[str, Any]] = {}
JUDGE_TASKS_CAP = 5


@app.post("/admin/judge", dependencies=PROTECTED_DEPENDENCIES)
def admin_judge_run(payload: JudgeRequest | None = None):
    """对最新一份 e2e 结果发起离线 LLM 评判。"""

    running = [t for t in JUDGE_TASKS.values() if t.get("status") == "running"]
    if running:
        raise HTTPException(status_code=409, detail="已有评判任务在运行，请等它结束。")
    e2e_path = _latest_eval_result("e2e")
    if e2e_path is None:
        raise HTTPException(status_code=404, detail="data/eval 下没有 e2e 结果文件可评判。")

    # 只留最近几次任务记录，防 dict 无限涨。
    if len(JUDGE_TASKS) >= JUDGE_TASKS_CAP:
        for task_id in sorted(JUDGE_TASKS, key=lambda k: JUDGE_TASKS[k].get("started", 0))[:-JUDGE_TASKS_CAP + 1]:
            JUDGE_TASKS.pop(task_id, None)

    task_id = uuid4().hex[:8]
    JUDGE_TASKS[task_id] = {
        "status": "running",
        "done": 0,
        "total": None,
        "started": time.time(),
    }

    limit = payload.limit if payload else None

    def _worker() -> None:
        def on_progress(done: int, total: int) -> None:
            JUDGE_TASKS[task_id]["done"] = done
            JUDGE_TASKS[task_id]["total"] = total

        try:
            from app.llm_judge import run_judge

            summary = run_judge(e2e_path, limit=limit, on_progress=on_progress)
            JUDGE_TASKS[task_id]["status"] = "done"
            JUDGE_TASKS[task_id]["summary"] = summary
        except Exception as exc:
            logger.exception("LLM 评判任务失败")
            JUDGE_TASKS[task_id]["status"] = "error"
            JUDGE_TASKS[task_id]["error"] = str(exc)[:300]

    threading.Thread(target=_worker, daemon=True, name="llm-judge").start()
    return {"task_id": task_id, "status": "running"}


@app.get("/admin/judge/{task_id}", dependencies=PROTECTED_DEPENDENCIES)
def admin_judge_status(task_id: str):
    task = JUDGE_TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在或已被清理。")
    return task


# --------------------------------------------------------------------------
# 论文增量入库：上传 PDF → 后台线程执行 → 前端轮询进度。
#
# MinerU 解析单篇要 1~3 分钟，HTTP 请求同步等会撞超时，所以做成任务：
# POST 提交立即返回 task_id，GET /admin/ingest/{task_id} 轮询日志与结果。
# 全局同时只允许一个入库任务（Milvus 删插与本地文件合并不是线程安全的）。

INGEST_TASKS: dict[str, dict[str, Any]] = {}
INGEST_TASKS_CAP = 20
_ingest_busy = threading.Lock()
PAPERS_DIR = PROJECT_DIR / "data" / "raw" / "papers"

# 飞书同步扫描只负责发现变化和排队，真正入库仍由独立 Worker 完成。
# API 触发放到短生命周期线程里，避免扫描大文件夹时卡住 HTTP 请求。
FEISHU_SYNC_TASK: dict[str, Any] = {
    "status": "idle",
    "started_at": None,
    "completed_at": None,
    "result": None,
    "error": None,
}
_feishu_sync_busy = threading.Lock()


def _open_sync_store():
    from app.feishu_sync.settings import FeishuSyncSettings
    from app.feishu_sync.store import SyncStore

    return SyncStore(FeishuSyncSettings.from_app_settings().db_path)


@app.get("/admin/feishu-sync/status", dependencies=PROTECTED_DEPENDENCIES)
def admin_feishu_sync_status():
    """查看同步源、最近扫描、任务积压和当前知识库版本。"""

    with _open_sync_store() as store:
        status = store.status()
    for source in status.get("sources", []):
        source.pop("tenant_key", None)
        source.pop("root_token", None)
    return {"success": True, "runner": dict(FEISHU_SYNC_TASK), **status}


@app.get("/admin/feishu-sync/documents", dependencies=PROTECTED_DEPENDENCIES)
def admin_feishu_sync_documents(
    status: str = Query("", max_length=40),
    limit: int = Query(100, ge=1, le=500),
):
    """列出飞书来源登记与最近同步状态，不返回完整 file token。"""

    with _open_sync_store() as store:
        documents = store.list_documents(limit=limit, status=status)
    for document in documents:
        document.pop("source_token", None)
        document.pop("tenant_key", None)
        document.pop("root_token", None)
        document.pop("folder_token", None)
        document.pop("parent_folder_token", None)
    return {"success": True, "documents": documents}


@app.get("/admin/feishu-sync/jobs", dependencies=PROTECTED_DEPENDENCIES)
def admin_feishu_sync_jobs(
    status: str = Query("", max_length=40),
    limit: int = Query(100, ge=1, le=500),
):
    """查看同步任务状态、重试次数和结构化错误。"""

    with _open_sync_store() as store:
        jobs = store.list_jobs(limit=limit, status=status)
    for job in jobs:
        job.pop("source_token", None)
        job.pop("lease_owner", None)
    return {"success": True, "jobs": jobs}


@app.post("/admin/feishu-sync/run", dependencies=PROTECTED_DEPENDENCIES)
def admin_feishu_sync_run():
    """后台触发一次完整递归对账；不会在 API 进程里执行入库。"""

    settings = get_settings()
    if not settings.feishu_sync_enabled:
        raise HTTPException(
            status_code=409,
            detail={
                "success": False,
                "error_type": "sync_disabled",
                "message": "飞书自动同步尚未启用。",
                "suggestion": "把 FEISHU_SYNC_ENABLED 设为 true 后重启服务。",
            },
        )
    if not _feishu_sync_busy.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="已有飞书扫描正在运行。")

    task_id = uuid4().hex[:12]
    FEISHU_SYNC_TASK.update(
        task_id=task_id,
        status="running",
        started_at=datetime.now().isoformat(timespec="seconds"),
        completed_at=None,
        result=None,
        error=None,
    )

    def task() -> None:
        from app.feishu_sync.cli import _drift_checker
        from app.feishu_sync.service import ReconcileService
        from app.feishu_sync.settings import FeishuSyncSettings

        service = None
        try:
            sync_settings = FeishuSyncSettings.from_app_settings()
            service = ReconcileService(sync_settings, drift_checker=_drift_checker())
            result = service.reconcile().as_dict()
            FEISHU_SYNC_TASK.update(status="completed", result=result)
        except Exception as exc:
            logger.exception("管理接口触发飞书对账失败")
            FEISHU_SYNC_TASK.update(
                status="failed", error=f"{type(exc).__name__}: {str(exc)[:1000]}"
            )
        finally:
            if service is not None:
                service.close()
            FEISHU_SYNC_TASK["completed_at"] = datetime.now().isoformat(
                timespec="seconds"
            )
            _feishu_sync_busy.release()

    threading.Thread(target=task, daemon=True, name=f"feishu-scan-{task_id}").start()
    return {"success": True, "task_id": task_id, "status": "running"}


@app.post(
    "/admin/feishu-sync/jobs/{job_id}/retry",
    dependencies=PROTECTED_DEPENDENCIES,
)
def admin_feishu_sync_retry(job_id: int):
    """把已失败任务重置为 pending，交给独立 Worker 重试。"""

    with _open_sync_store() as store:
        retried = store.retry_job(job_id)
    if not retried:
        raise HTTPException(status_code=404, detail="没有找到可重试的失败任务。")
    return {"success": True, "job_id": job_id, "status": "pending"}


@app.post("/admin/ingest", dependencies=PROTECTED_DEPENDENCIES)
async def admin_ingest(
    file: UploadFile = File(...),
    item_id: str = Form(""),
    title: str = Form(""),
    reader: str = Form(""),
    theme: str = Form(""),
    force: bool = Form(False),
):
    """上传论文 PDF 并启动后台增量入库。"""

    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_type": "bad_request",
                "message": "只接受 PDF 文件。",
                "suggestion": "上传 .pdf 后缀的论文文件。",
            },
        )
    if not _ingest_busy.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail={
                "success": False,
                "error_type": "busy",
                "message": "已有入库任务在运行。",
                "suggestion": "等当前任务结束后再提交。",
            },
        )

    save_path = PAPERS_DIR / filename
    content = await file.read()
    save_path.write_bytes(content)

    task_id = uuid4().hex[:12]
    INGEST_TASKS[task_id] = {
        "task_id": task_id,
        "status": "running",
        "item_id": item_id,
        "filename": filename,
        "logs": [],
        "report": None,
        "error": None,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }

    def task() -> None:
        sys.path.insert(0, str(PROJECT_DIR / "scripts"))
        from ingest_paper import IngestionError, run_ingestion

        try:
            def log(message: str) -> None:
                INGEST_TASKS[task_id]["logs"].append(str(message))

            report = run_ingestion(
                save_path,
                item_id=item_id or None,
                title=title,
                reader=reader,
                theme=theme,
                force=force,
                log=log,
            )
            INGEST_TASKS[task_id].update(status="done" if report["status"] == "ingested" else "skipped", report=report)
        except IngestionError as exc:
            INGEST_TASKS[task_id].update(status="failed", error=str(exc), failures=exc.failures)
        except Exception as exc:
            INGEST_TASKS[task_id].update(status="failed", error=f"{type(exc).__name__}：{exc}")
        finally:
            _ingest_busy.release()
            while len(INGEST_TASKS) > INGEST_TASKS_CAP:
                oldest = min(INGEST_TASKS, key=lambda key: INGEST_TASKS[key]["started_at"])
                INGEST_TASKS.pop(oldest, None)

    threading.Thread(target=task, daemon=True, name=f"ingest-{task_id}").start()
    return {"success": True, "task_id": task_id, "status": "running"}


@app.get("/admin/ingest/{task_id}", dependencies=PROTECTED_DEPENDENCIES)
def admin_ingest_status(task_id: str):
    """查询入库任务的状态、日志与报告。"""

    task = INGEST_TASKS.get(task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error_type": "not_found",
                "message": f"入库任务 {task_id} 不存在（服务重启后任务记录会清空）。",
                "suggestion": "重新提交入库请求。",
            },
        )
    return {"success": True, **task}


@app.post("/chat", dependencies=PROTECTED_DEPENDENCIES)
def chat(request: ChatRequest):
    """标准工具调用 Agent 接口。

    适合普通问答、开放式问题。
    模型自己决定调用哪个工具、调用几次、什么时候结束。
    """

    started_at = time.perf_counter()

    try:
        result = run_agent(
            get_tool_agent(),
            question=request.question,
            thread_id=request.thread_id,
        )
        messages = result.get("messages", [])
        answer = messages[-1].content if messages else ""
        trace = [simplify_message(message) for message in messages]
        trace_summary = build_trace_summary(messages, answer=answer)
        latency_ms = int((time.perf_counter() - started_at) * 1000)

        log_id = _safe_record_agent_log(
            mode="tool_agent",
            thread_id=request.thread_id,
            question=request.question,
            answer=answer,
            trace=trace,
            success=True,
            error="",
            latency_ms=latency_ms,
            intent=result["intent"].intent,
            token_usage=result.get("token_usage"),
        )
    except Exception as exc:
        log_failure_and_raise(
            mode="tool_agent",
            request=request,
            exc=exc,
            started_at=started_at,
        )

    rewrite = result["rewrite"]
    intent = result["intent"]
    response = {
        "mode": "tool_agent",
        "question": request.question,
        "thread_id": request.thread_id,
        "answer": answer,
        "log_id": log_id,
        "latency_ms": latency_ms,
        "original_question": rewrite.original_question,
        "rewritten_query": rewrite.rewritten_query,
        "resolved_item_id": rewrite.item_id,
        "rewrite_reason": rewrite.reason,
        "intent": intent.intent,
        "intent_reason": intent.reason,
        "token_usage": result.get("token_usage"),
    }

    if request.include_trace:
        response["trace"] = trace
        response["trace_summary"] = trace_summary

    return response


@app.post("/planner-chat", dependencies=PROTECTED_DEPENDENCIES)
def planner_chat(request: ChatRequest):
    """可控 PlannerAgent 接口。

    适合组会汇报提纲、论文对比、多资料综合这类复杂任务。
    它会先生成计划，再按计划调用检索工具。
    """

    started_at = time.perf_counter()

    try:
        result = get_planner_agent().answer(
            request.question,
            thread_id=request.thread_id,
        )
        raw_step_results = result["step_results"]
        step_results = [
            simplify_step_result(step_result)
            for step_result in result["step_results"]
        ]
        trace = {
            "plan": asdict(result["plan"]),
            "plan_text": format_plan(result["plan"]),
            "step_results": step_results,
        }
        latency_ms = int((time.perf_counter() - started_at) * 1000)

        log_id = _safe_record_agent_log(
            mode="planner_agent",
            thread_id=request.thread_id,
            question=request.question,
            answer=result["answer"],
            trace=trace,
            success=True,
            error="",
            latency_ms=latency_ms,
            intent=result["intent"].intent,
            token_usage=result.get("token_usage"),
        )
    except Exception as exc:
        log_failure_and_raise(
            mode="planner_agent",
            request=request,
            exc=exc,
            started_at=started_at,
        )

    rewrite = result["rewrite"]
    intent = result["intent"]
    response = {
        "mode": "planner_agent",
        "question": request.question,
        "thread_id": request.thread_id,
        "answer": result["answer"],
        "plan": trace["plan"],
        "plan_text": trace["plan_text"],
        "log_id": log_id,
        "latency_ms": latency_ms,
        "original_question": rewrite.original_question,
        "rewritten_query": rewrite.rewritten_query,
        "resolved_item_id": rewrite.item_id,
        "rewrite_reason": rewrite.reason,
        "intent": intent.intent,
        "intent_reason": intent.reason,
        "token_usage": result.get("token_usage"),
    }

    if request.include_trace:
        response["step_results"] = step_results

    # 来源与 /ask 同一套提取逻辑：前端路由到会话存储时要用。
    planner_sources = []
    for step_result in raw_step_results:
        planner_sources.extend(
            extract_sources_from_tool_text(
                step_result.get("tool", "tool"),
                step_result.get("result", ""),
            )
        )
    response["sources"] = order_sources_by_citation(
        dedupe_sources(planner_sources)
    )[:20]

    return response


@app.post("/ask", dependencies=PROTECTED_DEPENDENCIES)
def ask(request: ChatRequest):
    """统一入口：由意图识别自动选择执行链路。

    组会汇报提纲、文献对比这类步骤相对固定的任务交给 PlannerAgent，
    其余交给 Tool Agent，因为它的工具集更全。

    响应里的 mode 是实际执行的链路，intent 说明为什么这么选。
    预处理只做一次，两条链路共用同一份改写与分类结果。
    """

    started_at = time.perf_counter()
    planned_mode = TOOL_AGENT

    try:
        prepared = prepare_question(request.question, request.thread_id, mode="auto")
        planned_mode = route_for_intent(prepared.intent.intent)

        if planned_mode == PLANNER_AGENT:
            result = get_planner_agent().answer(
                request.question,
                thread_id=request.thread_id,
                prepared=prepared,
            )
            step_results = [
                simplify_step_result(step_result)
                for step_result in result["step_results"]
            ]
            plan = asdict(result["plan"])
            plan_text = format_plan(result["plan"])
            answer = result["answer"]
            latency_ms = int((time.perf_counter() - started_at) * 1000)

            log_id = _safe_record_agent_log(
                mode=PLANNER_AGENT,
                thread_id=request.thread_id,
                question=request.question,
                answer=answer,
                trace={
                    "plan": plan,
                    "plan_text": plan_text,
                    "step_results": step_results,
                },
                success=True,
                error="",
                latency_ms=latency_ms,
                intent=prepared.intent.intent,
                token_usage=result.get("token_usage"),
            )
        else:
            result = run_agent(
                get_tool_agent(),
                question=request.question,
                thread_id=request.thread_id,
                prepared=prepared,
            )
            messages = result.get("messages", [])
            answer = messages[-1].content if messages else ""
            trace = [simplify_message(message) for message in messages]
            trace_summary = build_trace_summary(messages, answer=answer)
            latency_ms = int((time.perf_counter() - started_at) * 1000)

            log_id = _safe_record_agent_log(
                mode=TOOL_AGENT,
                thread_id=request.thread_id,
                question=request.question,
                answer=answer,
                trace=trace,
                success=True,
                error="",
                latency_ms=latency_ms,
                intent=prepared.intent.intent,
                token_usage=result.get("token_usage"),
            )
    except Exception as exc:
        log_failure_and_raise(
            mode=planned_mode,
            request=request,
            exc=exc,
            started_at=started_at,
        )

    rewrite = result["rewrite"]
    intent = result["intent"]
    response = {
        "mode": planned_mode,
        "question": request.question,
        "thread_id": request.thread_id,
        "answer": answer,
        "log_id": log_id,
        "latency_ms": latency_ms,
        "original_question": rewrite.original_question,
        "rewritten_query": rewrite.rewritten_query,
        "resolved_item_id": rewrite.item_id,
        "rewrite_reason": rewrite.reason,
        "intent": intent.intent,
        "intent_reason": intent.reason,
        "token_usage": result.get("token_usage"),
    }

    if planned_mode == PLANNER_AGENT:
        response["plan"] = plan
        response["plan_text"] = plan_text
        if request.include_trace:
            response["step_results"] = step_results
        # Planner 链路的来源从每一步的检索原文里提——
        # 和流式路径同一套提取与排序，答案里的 [n] 能对上。
        planner_sources = []
        for step_result in result["step_results"]:
            planner_sources.extend(
                extract_sources_from_tool_text(
                    step_result.get("tool", "tool"),
                    step_result.get("result", ""),
                )
            )
        response["sources"] = cap_sources_with_citations(
            planner_sources, answer
        )
    elif request.include_trace:
        response["trace"] = trace
        response["trace_summary"] = trace_summary
        response["sources"] = trace_summary["sources"]

    return response


# SSE 队列桥接专用的线程池。
#
# 不能用 asyncio 默认线程池：消费端每次读队列最长会阻塞一个轮询周期，
# 默认池还要服务其它 run_in_executor 调用，并发流一多就会把池耗尽，
# 连累整个事件循环。
_stream_executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="sse-bridge")

# 消费端单次读队列的超时（秒）。拿不到事件就发一次心跳并检查客户端
# 是否已经断开，而不是无限期占住一个线程。
_STREAM_POLL_SECONDS = 15.0


@app.post("/ask/stream", dependencies=PROTECTED_DEPENDENCIES)
async def ask_stream(http_request: Request, request: ChatRequest):
    """流式版的 /ask。

    一边跑一边推：理解结果、正在调用的工具、答案的每个片段。

    为什么值得做：一次问答要四秒上下，其中大半时间在做检索和重排。
    同步接口下用户看到的就是四秒空白，然后整段答案蹦出来。流式把这些
    中间步骤露出来，同样的耗时感觉短得多。

    实现上 Agent 跑在独立线程里，用队列把事件桥接给异步生成器——
    它的 checkpointer 不支持异步方法。详见 app/streaming.py。
    """

    events: queue.Queue = queue.Queue()
    sentinel = object()
    # 客户端断开后由消费端置位，worker 在步骤间隙检查它，尽早止损。
    cancelled = threading.Event()
    # 收集工具结果原文，流式结束后提取「依据来源」放进 done 事件。
    tool_texts: list[tuple[str, str]] = []

    def worker() -> None:
        started_at = time.perf_counter()
        parts: list[str] = []
        tools: list[dict] = []
        answer = ""
        mode = TOOL_AGENT
        prepared = None

        try:
            prepared = prepare_question(request.question, request.thread_id, mode="auto")
            mode = route_for_intent(prepared.intent.intent)

            # Tool Agent 链路：先自愈断连留下的悬空工具调用，再按会话
            # 历史续号（新编号接在上一回合最大「资料N」之后，跨回合不
            # 撞号）；history_tool_texts 供结束后合并来源。Planner 无
            # checkpointer，跳过这两步。
            history_tool_texts: list[tuple[str, str]] = []
            if mode == TOOL_AGENT:
                heal_dangling_tool_history(get_tool_agent(), prepared.config)
                history_tool_texts = seed_registry_from_history(
                    get_tool_agent(), prepared.config
                )

            events.put(
                StreamEvent(
                    "stage",
                    {
                        "stage": "understood",
                        "rewritten_query": prepared.rewrite.rewritten_query,
                        "rewritten": prepared.rewrite.rewritten,
                        "item_id": prepared.rewrite.item_id,
                        "intent": prepared.intent.intent,
                        "mode": mode,
                    },
                )
            )

            if mode == PLANNER_AGENT:
                # Planner 的计划与检索没有 token 可流，但两件事可以做：
                # ① 每步检索完成实时推 tool 事件（工具面板逐条亮起来）；
                # ② 最终答案走 llm.stream，逐段推给前端。
                planner = get_planner_agent()

                def on_step(step_result: dict) -> None:
                    events.put(
                        StreamEvent(
                            "tool",
                            {
                                "name": str(step_result.get("tool", "tool")),
                                "query": str(step_result.get("query", "")),
                            },
                        )
                    )

                run = planner.prepare_execution(
                    request.question,
                    thread_id=request.thread_id,
                    prepared=prepared,
                    on_step=on_step,
                )
                for step_result in run.step_results:
                    tools.append(
                        {
                            "name": str(step_result.get("tool", "tool")),
                            "query": str(step_result.get("query", "")),
                        }
                    )
                    tool_texts.append(
                        (
                            str(step_result.get("tool", "tool")),
                            str(step_result.get("result", "")),
                        )
                    )

                for chunk in planner.llm.stream(run.prompt, config=run.prepared.config):
                    if cancelled.is_set():
                        logger.info("客户端断开，中止本次流式问答。")
                        return
                    text = str(getattr(chunk, "content", "") or "")
                    if text:
                        parts.append(text)
                        events.put(StreamEvent("token", {"content": text}))
                answer = "".join(parts)
                planner.record_turn(run, answer)
            else:
                for event in iter_agent_events(
                    get_tool_agent(),
                    HumanMessage(content=prepared.rewrite.rewritten_query),
                    prepared.config,
                ):
                    if cancelled.is_set():
                        # 客户端已经断开，剩下的检索和生成没有意义。
                        # 会话历史与请求日志也不再补写——这轮对话没有完成。
                        logger.info("客户端断开，中止本次流式问答。")
                        return
                    if event.type == "token":
                        parts.append(str(event.data.get("content", "")))
                    elif event.type == "tool":
                        tools.append(event.data)
                        # 工具调用一来，说明刚累积的 token 是 ReAct 中间回合的
                        # 过渡文本（“我先查一下…”），不是最终答案——清空，
                        # answer 只保留最后一个回合的正式回答。
                        parts.clear()
                    elif event.type == "tool_result":
                        tool_texts.append(
                            (str(event.data.get("name", "tool")), str(event.data.get("text", "")))
                        )
                    events.put(event)
                answer = "".join(parts)

            append_turn(
                request.thread_id,
                Turn(
                    question=request.question,
                    rewritten_query=prepared.rewrite.rewritten_query,
                    item_id=prepared.rewrite.item_id,
                    answer=answer,
                    intent=prepared.intent.intent,
                ),
                settings=prepared.settings,
            )

            latency_ms = int((time.perf_counter() - started_at) * 1000)

            # 依据来源：本轮工具结果 + 会话历史最近回合的工具结果。
            # 历史那部分是必须的——编号按会话续接后，模型引的历史号
            # （比如复用上一轮检索作答时）也能在面板里对上。
            # 排序按编号升序，答案里的 [n] 点过来能对上。
            sources = []
            for tool_name, text in tool_texts:
                sources.extend(extract_sources_from_tool_text(tool_name, text))
            if mode == TOOL_AGENT:
                for tool_name, text in history_tool_texts:
                    sources.extend(extract_sources_from_tool_text(tool_name, text))

            unique_sources = cap_sources_with_citations(sources, answer)

            _safe_record_agent_log(
                mode=mode,
                thread_id=request.thread_id,
                question=request.question,
                answer=answer,
                trace={"tools": tools, "streamed": True},
                success=True,
                error="",
                latency_ms=latency_ms,
                intent=prepared.intent.intent,
                token_usage=read_token_usage(prepared.config),
            )
            events.put(
                StreamEvent(
                    "done",
                    {
                        "answer": answer,
                        "mode": mode,
                        "intent": prepared.intent.intent,
                        "original_question": prepared.rewrite.original_question,
                        "rewritten_query": prepared.rewrite.rewritten_query,
                        "tools": tools,
                        "sources": unique_sources,
                        "latency_ms": latency_ms,
                        "token_usage": read_token_usage(prepared.config),
                    },
                )
            )

        except Exception as exc:
            failure = classify_exception(exc)
            logger.exception("流式问答失败：%s", exc)
            try:
                record_agent_log(
                    mode=mode,
                    thread_id=request.thread_id,
                    question=request.question,
                    answer=answer,
                    trace={"tools": tools, "streamed": True},
                    success=False,
                    error=str(exc),
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    # prepared 为 None 说明预处理阶段就挂了，那时还没有
                    # 计量器与意图可记。
                    intent=prepared.intent.intent if prepared else None,
                    token_usage=read_token_usage(prepared.config) if prepared else None,
                )
            except Exception:
                logger.warning("失败日志也没写进去。")
            events.put(
                StreamEvent(
                    "error",
                    {
                        # classify_exception 返回的是 dataclass，不是 dict。
                        "message": failure.message,
                        "error_type": failure.error_type,
                        "suggestion": failure.suggestion,
                    },
                )
            )
        finally:
            events.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()

    async def generate():
        try:
            while True:
                # 带超时地轮询队列：等待期间定期发心跳、检查客户端是否还在，
                # 而不是无限期阻塞一个线程直到 worker 结束。
                try:
                    event = await asyncio.get_running_loop().run_in_executor(
                        _stream_executor,
                        lambda: events.get(timeout=_STREAM_POLL_SECONDS),
                    )
                except queue.Empty:
                    if await http_request.is_disconnected():
                        cancelled.set()
                        logger.info("客户端在等待期间断开，停止推送 SSE 事件。")
                        break
                    # SSE 注释行，浏览器和 EventSource 会忽略，反代却会因此
                    # 重置空闲计时器，长检索期间连接不至于被掐断。
                    yield ": keep-alive\n\n"
                    continue

                if event is sentinel:
                    break
                yield f"data: {json.dumps(event.to_payload(), ensure_ascii=False)}\n\n"
        finally:
            # 客户端在 token 流期间断开时，Starlette 会直接关闭这个生成器
            # （GeneratorExit 从 yield 处抛出），上面轮询路径检查不到——
            # 所以在这里统一置位取消标志，worker 在步骤间隙看到就止损。
            # 正常结束时 worker 已经跑完，置位是无害的空操作。
            cancelled.set()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 关掉 nginx 一类反代的缓冲，否则流会被攒成一坨再发。
            "X-Accel-Buffering": "no",
        },
    )
