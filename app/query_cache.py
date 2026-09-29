import hashlib
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path

from app.config import get_settings


# 进程内查询结果缓存：TTL + LRU，与内存限流（app/rate_limit.py）同一风格——
# 单进程假设、无外部依赖、锁只保护字典操作不保护慢检索。
#
# 缓的是 _paper_retrieve 的完整产出（翻译→嵌入→检索→重排→父块扩展之后的
# 三元组）。实测一次全链路约 4 秒 + 全部 API 费用，相同问题命中时亚毫秒返回。
#
# TTL 防过期（语料重建后旧答案不该继续返回）；LRU 防膨胀（256 条约 2MB，
# 相比 BM25 索引的 379MB 可忽略）。服务重启缓存清空——可接受，语料是静态的。
#
# 注意：跑评测（scripts/run_eval.py）前要设 QUERY_CACHE_ENABLED=false，
# 否则一小时内的重跑会命中缓存，测不到检索层的真实变化。


_lock = threading.Lock()
_store: OrderedDict[str, tuple[float, object]] = OrderedDict()


def normalize_query(query: str) -> str:
    """归一化查询文本：折叠首尾与内部连续空白。

    刻意不做大小写归一（中文场景无意义）和标点归一（改动标点通常
    意味着问题真的变了，宁可漏缓存不可错缓存）。
    """

    return " ".join(str(query or "").split())


def make_key(
    query: str,
    item_id: str | None = None,
    min_score: float | None = None,
    top_k: int | None = None,
    corpus_revision: int | None = None,
    include_reference: bool | None = None,
) -> str:
    """组合缓存键。min_score 必须入键：意图分档下同一查询在事实档和
    总结档会得到不同的置信度判断，混用会串。include_reference 同理：
    内容型意图过滤参考文献、元数据类保留，同一查询两种过滤结果不同。"""

    if corpus_revision is None:
        corpus_revision = current_corpus_revision()
    raw = "\x1f".join(
        [
            normalize_query(query),
            str(item_id or ""),
            "" if min_score is None else f"{min_score:.4f}",
            str(top_k or ""),
            str(corpus_revision),
            "" if include_reference is None else str(include_reference),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def current_corpus_revision() -> int:
    """只读同步库的语料版本；库不存在或尚未迁移时沿用版本 0。"""

    raw_path = Path(get_settings().feishu_sync_db_path)
    if not raw_path.is_absolute():
        from app.config import PROJECT_DIR

        raw_path = PROJECT_DIR / raw_path
    if not raw_path.exists():
        return 0
    try:
        connection = sqlite3.connect(
            f"file:{raw_path.as_posix()}?mode=ro", uri=True, timeout=0.2
        )
        try:
            row = connection.execute(
                "SELECT revision FROM corpus_state WHERE singleton=1"
            ).fetchone()
            return int(row[0]) if row else 0
        finally:
            connection.close()
    except (sqlite3.Error, OSError, ValueError):
        return 0


def get(key: str):
    """取缓存。命中则刷新 LRU 顺序并检查 TTL；过期/不存在返回 None。"""

    settings = get_settings()
    now = time.monotonic()

    with _lock:
        entry = _store.get(key)
        if entry is None:
            return None
        written_at, value = entry
        if now - written_at >= settings.query_cache_ttl_seconds:
            _store.pop(key, None)
            return None
        _store.move_to_end(key)
        return value


def put(key: str, value) -> None:
    """写缓存，超过容量时淘汰最久未使用的条目。"""

    settings = get_settings()
    max_entries = max(1, settings.query_cache_max_entries)

    with _lock:
        _store[key] = (time.monotonic(), value)
        _store.move_to_end(key)
        while len(_store) > max_entries:
            _store.popitem(last=False)


def clear() -> None:
    """清空缓存。测试和「语料重建后立刻生效」场景用。"""

    with _lock:
        _store.clear()


def stats() -> dict:
    """当前状态（容量、条数），给运维和调试看。"""

    with _lock:
        return {"entries": len(_store)}
