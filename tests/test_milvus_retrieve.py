from app import milvus_store as ms
from app.milvus_store import MilvusStore, RetrievedChunk


# retrieve() 的编排逻辑：召回 → 重排 → 判断置信度。
#
# 这里不连 Milvus，用 object.__new__ 造一个空壳，把 hybrid_search 和
# rerank_documents 都替换掉，单独验这段编排。
#
# 最要紧的一条是降级：重排挂了必须退回原始顺序并且**不判置信度**。
# 如果重排失败时默认判成「不置信」，那重排服务一抖，整个系统就对
# 所有问题回答「没找到」——比不重排还糟。


def make_store(settings, **overrides):
    store = object.__new__(MilvusStore)
    store.settings = settings
    for key, value in overrides.items():
        setattr(store, key, value)
    return store


def chunk(chunk_id: str, text: str = "内容") -> RetrievedChunk:
    return RetrievedChunk(score=0.5, text=text, metadata={"chunk_id": chunk_id})


def patch_retrieval(monkeypatch, store, chunks, ranked=None, calls=None):
    def fake_hybrid(*args, **kwargs):
        if calls is not None:
            calls["hybrid"] = kwargs
        return chunks

    def fake_rerank(query, documents, settings=None, **kwargs):
        if calls is not None:
            calls["rerank_query"] = query
            calls["rerank_documents"] = documents
        return ranked

    monkeypatch.setattr(store, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(ms, "rerank_documents", fake_rerank)


def test_confident_when_top_score_clears_threshold(monkeypatch, make_settings):
    store = make_store(make_settings(rerank_min_score=0.8))
    patch_retrieval(
        monkeypatch,
        store,
        [chunk("a"), chunk("b")],
        ranked=[(1, 0.95), (0, 0.31)],
    )

    result = store.retrieve("问题", top_k=2)

    assert result.confident is True
    assert result.reranked is True
    assert result.top_score == 0.95
    # 重排后要按新顺序返回，不是原来的顺序
    assert [c.metadata["chunk_id"] for c in result.chunks] == ["b", "a"]


def test_not_confident_when_top_score_below_threshold(monkeypatch, make_settings):
    store = make_store(make_settings(rerank_min_score=0.8))
    patch_retrieval(monkeypatch, store, [chunk("a")], ranked=[(0, 0.42)])

    result = store.retrieve("问题", top_k=1)

    assert result.confident is False
    assert result.top_score == 0.42
    # 仍然把结果带回来——由调用方决定怎么用，检索层不替它清空
    assert len(result.chunks) == 1


def test_not_confident_when_nothing_retrieved(make_settings):
    store = make_store(make_settings())
    store.hybrid_search = lambda *a, **k: []

    result = store.retrieve("问题", top_k=5)

    assert result.confident is False
    assert result.chunks == []
    assert result.reranked is False


def test_rerank_failure_does_not_judge_confidence(monkeypatch, make_settings):
    """重排挂了要退回原顺序，且不判置信度。

    这里若判成「不置信」，重排服务一抖整个系统就对所有问题说
    「没找到」，比不重排还糟。
    """

    store = make_store(make_settings(rerank_enabled=True))
    patch_retrieval(monkeypatch, store, [chunk("a"), chunk("b")], ranked=None)

    result = store.retrieve("问题", top_k=2)

    assert result.confident is True
    assert result.reranked is False
    assert result.top_score is None
    assert [c.metadata["chunk_id"] for c in result.chunks] == ["a", "b"]


def test_rerank_disabled_skips_the_call(monkeypatch, make_settings):
    store = make_store(make_settings(rerank_enabled=False))
    calls = {}
    patch_retrieval(monkeypatch, store, [chunk("a")], ranked=[(0, 0.99)], calls=calls)

    result = store.retrieve("问题", top_k=1)

    assert "rerank_documents" not in calls
    assert result.reranked is False
    assert result.confident is True


def test_explicit_rerank_flag_overrides_settings(monkeypatch, make_settings):
    """调用方显式传的 rerank 应当盖过配置。"""

    store = make_store(make_settings(rerank_enabled=True))
    calls = {}
    patch_retrieval(monkeypatch, store, [chunk("a")], ranked=None, calls=calls)

    result = store.retrieve("问题", top_k=1, rerank=False)

    assert "rerank_documents" not in calls
    assert result.reranked is False


def test_recalls_more_candidates_than_requested(monkeypatch, make_settings):
    """重排要有得挑，所以融合阶段必须多召回一些。"""

    store = make_store(make_settings(rerank_enabled=True))
    calls = {}
    patch_retrieval(monkeypatch, store, [chunk("a")], ranked=[(0, 0.9)], calls=calls)

    store.retrieve("问题", top_k=3, recall_limit=20)

    assert calls["hybrid"]["top_k"] == 20
    assert calls["hybrid"]["recall_limit"] == 20


def test_without_rerank_asks_for_exactly_top_k(monkeypatch, make_settings):
    store = make_store(make_settings(rerank_enabled=False))
    calls = {}
    patch_retrieval(monkeypatch, store, [chunk("a")], ranked=None, calls=calls)

    store.retrieve("问题", top_k=3)

    assert calls["hybrid"]["top_k"] == 3


def test_truncates_to_top_k_after_rerank(monkeypatch, make_settings):
    store = make_store(make_settings(rerank_enabled=True))
    ranked = [(index, 1.0 - index / 100) for index in range(10)]
    patch_retrieval(monkeypatch, store, [chunk(f"c{i}") for i in range(10)], ranked=ranked)

    result = store.retrieve("问题", top_k=3, recall_limit=20)

    assert len(result.chunks) == 3
    assert [c.metadata["chunk_id"] for c in result.chunks] == ["c0", "c1", "c2"]


def test_passes_original_question_to_rerank(monkeypatch, make_settings):
    """交给重排的是用户原话，不是翻译后的英文查询。

    重排模型是多语言的，中文问题配英文文档它处理得了；而翻译后的
    查询丢了原问题的措辞，反而判不准「这段能不能回答用户问的」。
    """

    store = make_store(make_settings())
    calls = {}
    patch_retrieval(monkeypatch, store, [chunk("a")], ranked=[(0, 0.9)], calls=calls)

    store.retrieve("生计韧性怎么衡量", top_k=1)

    assert calls["rerank_query"] == "生计韧性怎么衡量"
