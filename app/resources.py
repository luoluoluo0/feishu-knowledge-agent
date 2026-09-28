import logging
import time
from functools import lru_cache

from app.tools import PaperSearchTools


# 进程级共享的检索栈。
#
# PaperSearchTools 的构造是全项目最重的初始化：一条 Milvus gRPC 连接 +
# 把 13.5MB 的 chunks.jsonl 全量读入、逐条分词建 BM25 索引——实测单实例
# 纯构建约 10 秒、379MB 堆内存。之前 Tool Agent 和 PlannerAgent 各自
# new 一套（内容完全相同的两份），GET /tools 还要每次请求临时再造一套。
#
# 检索栈在建成后是只读的（连接查询、内存索引查询），天然并发安全，
# 现在缓存 Agent 本来就在多个并发请求间共享它，所以进程内一份就够。
# 两个 Agent 检索行为的差异在「谁决定查什么」，不在基础设施。

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_retrieval_stack() -> PaperSearchTools:
    """拿到进程级检索栈单例，首次调用时构建（约 10 秒）。"""

    started = time.perf_counter()
    stack = PaperSearchTools()
    logger.info(
        "检索栈初始化完成，耗时 %.1f 秒（Milvus 连接 + BM25 索引加载）",
        time.perf_counter() - started,
    )
    return stack


def close_retrieval_stack() -> None:
    """释放共享检索栈：关闭 Milvus 连接并清掉单例缓存。幂等。"""

    if get_retrieval_stack.cache_info().currsize == 0:
        return

    stack = get_retrieval_stack()  # 已缓存，不会触发重建
    try:
        stack.paper_store.client.close()
    except Exception:
        logger.warning("关闭 Milvus 连接失败（进程即将退出，忽略）：", exc_info=True)

    get_retrieval_stack.cache_clear()
    logger.info("检索栈已释放。")
