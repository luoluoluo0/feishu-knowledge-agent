import json
import logging
import urllib.error
import urllib.request

from app.config import Settings, get_settings


# rerank 客户端。
#
# 检索分两段：召回（粗）和重排（精）。召回用双塔模型——问题和文档
# 分别编码成向量再算余弦，快，但糙，因为它从来没见过「问题 + 文档」
# 这一对。重排用交叉编码器，把两者拼在一起过一遍模型，回答的是
# 「这段文字能不能回答这个问题」，而不是「两者像不像」。
#
# 实测差别（300 条测试集，召回 20 条后整批重排）：
#
#   双塔余弦   该命中的 0.59 / 语料外的 0.55    完全重叠，切不开
#   rerank     该命中的 中位 0.995               分得开
#              语料外的 中位 0.053
#
# 重排顺带把检索质量也提上去了：原始 top-1 命中期望块 45.8%，
# 重排后 73.5%，+27.7pp。
#
# 所以它一次解决两件事：精排，以及给置信度提供一条能画的分数线。


logger = logging.getLogger(__name__)

# 模型名、门槛值、开关都从 Settings 读，默认值定义在 app/config.py，
# 这里不再放一份，免得两处不一致。

# 送进 rerank 的单条文档截断长度。
#
# 判断「这段能不能回答问题」看开头一段就够，长文只会拖慢速度。
# 整批 20 条 × 1200 字仍在模型接受范围内。
DOC_MAX_CHARS = 1200


def rerank_documents(
    query: str,
    documents: list[str],
    settings: Settings | None = None,
    timeout: int = 90,
) -> list[tuple[int, float]] | None:
    """把候选整批交给 rerank，返回 (原下标, 相关性分)，按分降序。

    一次送多条而不是只送一条。重排器的价值在于「从一批候选里把最相关的
    挑上来」——只喂一条等于问它「这段相关吗」，丢掉了它最擅长的部分。
    早期就踩过这个坑：只送 top-1 时，本该命中却拿不到期望块的那些题
    全落在 0.000 附近，看起来像分不开，其实是方法用错了。

    失败一律返回 None，由调用方决定怎么退。重排是锦上添花，不该因为
    它挂了就让整条检索链路失败。
    """

    if not query or not documents:
        return None

    settings = settings or get_settings()
    if not settings.silicon_api_key:
        logger.warning("没有配置 SILICON_API_KEY，跳过重排。")
        return None

    body = {
        "model": settings.rerank_model,
        "query": query,
        "documents": [str(doc)[:DOC_MAX_CHARS] for doc in documents],
        "top_n": len(documents),
    }
    request = urllib.request.Request(
        settings.embeddings_base_url.rstrip("/") + "/rerank",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.silicon_api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:160]
        logger.warning("重排接口返回 HTTP %s：%s", exc.code, detail)
        return None
    except Exception as exc:
        logger.warning("重排调用失败，退回原始顺序：%s", exc)
        return None

    results = payload.get("results") or []
    if not results:
        logger.warning("重排接口没有返回结果。")
        return None

    try:
        ranked = [
            (int(item.get("index", 0)), float(item.get("relevance_score", 0.0)))
            for item in results
        ]
    except (TypeError, ValueError) as exc:
        logger.warning("重排返回的格式不对：%s", exc)
        return None

    # 只保留下标合法的，防止接口返回越界下标时下游取错文档。
    ranked = [pair for pair in ranked if 0 <= pair[0] < len(documents)]
    if not ranked:
        return None

    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked
