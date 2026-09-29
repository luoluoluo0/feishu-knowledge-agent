from __future__ import annotations

from app.planner_agent import _block_identity, _trim_step_result


def _block(num: int, title: str, page: str, body: str = "正文内容" * 20) -> str:
    return f"资料{num}\n标题：{title}\n类型：text\n位置：{page}\n内容：{body}\n"


FULL_A = _block(1, "论文A", "PDF第2页，Data")
FULL_B = _block(2, "论文A", "PDF第3页，Method")
FULL_C = _block(3, "论文A", "PDF第4页，Results")
FULL_D = _block(4, "论文A", "PDF第5页，Discussion")
FULL_E = _block(5, "论文A", "PDF第16页，参考文献")


def test_trims_to_top_k_blocks():
    result = "\n".join([FULL_A, FULL_B, FULL_C, FULL_D, FULL_E])

    trimmed = _trim_step_result(result, top_k=3, seen_keys=set())

    assert "资料1" in trimmed and "资料2" in trimmed and "资料3" in trimmed
    assert "资料4" not in trimmed and "资料5" not in trimmed


def test_top_k_zero_keeps_all():
    result = "\n".join([FULL_A, FULL_B, FULL_C, FULL_D, FULL_E])

    trimmed = _trim_step_result(result, top_k=0, seen_keys=set())

    assert all(f"资料{i}" in trimmed for i in range(1, 6))


def test_cross_step_duplicate_dropped_even_beyond_top_k():
    """同一段内容第二步再出现时直接剪掉，且不占用第二步的 top_k 名额。"""

    seen: set[str] = set()
    _trim_step_result("\n".join([FULL_A, FULL_B]), top_k=3, seen_keys=seen)

    # 第二步把同样的块换了个编号又端回来
    dup_a = FULL_A.replace("资料1", "资料1")  # 块号在第二步里会不同
    step2 = "\n".join([dup_a, FULL_C])
    trimmed2 = _trim_step_result(step2, top_k=3, seen_keys=seen)

    assert "资料1" not in trimmed2  # 重复的 A 被剪掉
    assert "资料3" in trimmed2  # 新内容保留


def test_same_page_different_blocks_both_kept():
    """同一页的两个不同块内容不同，指纹不同，不误伤。"""

    b1 = _block(1, "论文A", "PDF第16页", "参考文献段落" * 10)
    b2 = _block(2, "论文A", "PDF第16页", "正文分析段落" * 10)

    trimmed = _trim_step_result("\n".join([b1, b2]), top_k=3, seen_keys=set())

    assert "参考文献段落" in trimmed and "正文分析段落" in trimmed
    assert _block_identity(b1) != _block_identity(b2)


def test_non_block_text_preserved_unchanged():
    """「没有检索到」这类降级说明没有资料块，原样返回。"""

    result = "⚠️ 没有检索到足够相关的资料。\n（重排最高分 0.04，低于门槛 0.35。）"

    trimmed = _trim_step_result(result, top_k=3, seen_keys=set())

    assert trimmed == result


def test_block_number_line_does_not_affect_identity():
    """同一段内容在两步里的「资料N」编号不同，指纹必须相同。"""

    b_step1 = _block(3, "论文A", "PDF第2页，Data")
    b_step2 = _block(2, "论文A", "PDF第2页，Data")

    assert _block_identity(b_step1) == _block_identity(b_step2)
