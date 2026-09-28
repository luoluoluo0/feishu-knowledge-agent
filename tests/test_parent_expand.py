from app.milvus_store import (
    RetrievedChunk,
    expand_chunks_to_parents,
    parent_context_window,
)


def make_chunk(text: str, parent_id: str | None, score: float = 0.9) -> RetrievedChunk:
    metadata = {"title": "测试标题", "item_id": "001"}
    if parent_id is not None:
        metadata["parent_id"] = parent_id
    return RetrievedChunk(score=score, text=text, metadata=metadata)


class TestParentContextWindow:
    def test_short_parent_returned_as_is(self):
        assert parent_context_window("短文本", "短文本", 100) == "短文本"

    def test_long_parent_window_centers_on_child(self):
        parent = "前" * 2000 + "命中段落在这里" + "后" * 2000
        window = parent_context_window(parent, "命中段落在这里", 1000)
        assert len(window) <= 1004  # 正文 1000 + 前后各一个省略号（2 字）
        assert window.startswith("……")
        assert window.endswith("……")
        assert "命中段落在这里" in window

    def test_child_not_found_falls_back_to_head(self):
        parent = "A" * 3000
        window = parent_context_window(parent, "不存在的子块", 500)
        assert window == "A" * 500 + "……"

    def test_window_at_document_start_has_no_leading_ellipsis(self):
        parent = "命中段落在开头" + "B" * 2000
        window = parent_context_window(parent, "命中段落在开头", 1000)
        assert not window.startswith("……")
        assert window.endswith("……")


class TestExpandChunksToParents:
    def _lookup(self, parents: dict):
        return lambda parent_id: parents.get(parent_id)

    def test_child_replaced_by_parent_text(self):
        parents = {"001_p1": {"text": "父块的完整上下文" * 10}}
        chunks = [make_chunk("子块片段", "001_p1")]
        result = expand_chunks_to_parents(chunks, self._lookup(parents), 1600)
        assert len(result) == 1
        assert result[0].text == "父块的完整上下文" * 10
        assert result[0].metadata["parent_expanded"] is True
        assert result[0].metadata["title"] == "测试标题"

    def test_score_preserved_from_child_hit(self):
        parents = {"001_p1": {"text": "父块文本"}}
        chunks = [make_chunk("子块", "001_p1", score=0.87)]
        result = expand_chunks_to_parents(chunks, self._lookup(parents), 1600)
        assert result[0].score == 0.87

    def test_same_parent_hits_deduplicated(self):
        parents = {"001_p1": {"text": "父块文本"}}
        chunks = [
            make_chunk("第一个命中", "001_p1", score=0.9),
            make_chunk("第二个命中", "001_p1", score=0.8),
        ]
        result = expand_chunks_to_parents(chunks, self._lookup(parents), 1600)
        assert len(result) == 1
        assert result[0].text == "父块文本"

    def test_different_parents_all_kept(self):
        parents = {
            "001_p1": {"text": "父块一"},
            "001_p2": {"text": "父块二"},
        }
        chunks = [make_chunk("命中一", "001_p1"), make_chunk("命中二", "001_p2")]
        result = expand_chunks_to_parents(chunks, self._lookup(parents), 1600)
        assert len(result) == 2

    def test_missing_parent_keeps_child_chunk(self):
        chunks = [make_chunk("孤儿子块", "999_p9")]
        result = expand_chunks_to_parents(chunks, self._lookup({}), 1600)
        assert len(result) == 1
        assert result[0].text == "孤儿子块"
        assert "parent_expanded" not in result[0].metadata

    def test_chunk_without_parent_id_passes_through(self):
        chunks = [make_chunk("无父块信息", None)]
        result = expand_chunks_to_parents(chunks, self._lookup({}), 1600)
        assert len(result) == 1
        assert result[0].text == "无父块信息"

    def test_long_parent_window_budget_respected(self):
        parents = {"001_p1": {"text": "长" * 50000}}
        chunks = [make_chunk("短命中", "001_p1")]
        result = expand_chunks_to_parents(chunks, self._lookup(parents), 1600)
        assert len(result[0].text) <= 1602

    def test_rank_order_preserved(self):
        parents = {"p1": {"text": "父一"}, "p2": {"text": "父二"}, "p3": {"text": "父三"}}
        chunks = [
            make_chunk("命中一", "p1", score=0.9),
            make_chunk("命中二", "p2", score=0.8),
            make_chunk("命中三", "p3", score=0.7),
        ]
        result = expand_chunks_to_parents(chunks, self._lookup(parents), 1600)
        assert [c.text for c in result] == ["父一", "父二", "父三"]
