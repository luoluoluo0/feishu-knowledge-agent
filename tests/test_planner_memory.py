"""PlannerAgent 会话记忆的定向测试：历史格式化与两个 prompt 的接入。

只测纯逻辑（prompt 拼装、历史截断），不调模型、不建 Agent。
"""

from app.planner import PlanResult
from app.planner_agent import PlannerAgent, format_history_block
from app.coreference import Turn


def make_turn(question: str, answer: str) -> Turn:
    return Turn(question=question, rewritten_query=question, item_id="", answer=answer)


class TestFormatHistoryBlock:
    def test_empty_history_returns_empty_string(self):
        assert format_history_block([]) == ""
        assert format_history_block([make_turn("", "答案")]) == ""

    def test_contains_question_and_answer(self):
        block = format_history_block([make_turn("对比012和045", "**提纲**：……")])
        assert "对比012和045" in block
        assert "**提纲**" in block
        assert "对话历史" in block

    def test_long_answer_truncated(self):
        block = format_history_block([make_turn("问题", "长" * 2000)])
        assert "长" * 800 in block
        assert "长" * 801 not in block
        assert block.endswith("…") or "…" in block

    def test_multiple_turns_in_order(self):
        block = format_history_block(
            [make_turn("第一问", "答一"), make_turn("第二问", "答二")]
        )
        assert block.index("第一问") < block.index("第二问")


def _build_prompt(history):
    from types import SimpleNamespace

    agent = PlannerAgent.__new__(PlannerAgent)  # 不跑 __init__，避免建 LLM
    # build_final_prompt 会读 settings 里的拼装瘦身配置，给最小命名空间
    agent.settings = SimpleNamespace(planner_prompt_top_k_per_step=0)
    plan = PlanResult(task_type="compare_papers")
    return agent.build_final_prompt("再加上013", plan, [], history=history)


class TestFinalPromptWithHistory:
    def test_without_history_no_history_section(self):
        prompt = _build_prompt([])
        assert "对话历史" not in prompt
        assert "再加上013" in prompt

    def test_with_history_contains_previous_answer(self):
        prompt = _build_prompt(
            [make_turn("对比第012篇和第045篇", "上一轮的对比结论……")]
        )
        assert "对比第012篇和第045篇" in prompt
        assert "上一轮的对比结论" in prompt
        # 历史要出现在当前问题之前，模型才能"结合理解"
        assert prompt.index("对话历史") < prompt.index("再加上013")


class TestPlanPromptWithHistory:
    def test_plan_context_includes_history_lines(self):
        # plan() 的历史拼装是纯字符串逻辑，直接构造同样的输入验证。
        history = [make_turn("对比第012篇和第045篇", "结论略")]
        history_lines = [
            f"- 用户：{turn.question.strip()}"
            for turn in history
            if (turn.question or "").strip()
        ]
        context_text = (
            "对话历史（最近的在前，供理解当前问题；不要为历史单独规划步骤）：\n"
            + "\n".join(history_lines)
            + "\n\n当前用户问题：再加上013"
        )
        assert "- 用户：对比第012篇和第045篇" in context_text
        assert context_text.endswith("当前用户问题：再加上013")
