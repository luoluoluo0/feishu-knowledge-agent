import types

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app import context as context_module
from app.context import (
    build_context_hook,
    estimate_tokens,
    window_messages,
)


def _turn(human: str, tool_text: str, answer: str) -> list:
    """构造一轮标准对话：用户消息 → 模型带工具调用 → 工具结果 → 回答。"""

    return [
        HumanMessage(content=human),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_paper", "args": {"query": human}, "id": f"call_{human}"}],
        ),
        ToolMessage(content=tool_text, tool_call_id=f"call_{human}"),
        AIMessage(content=answer),
    ]


def _big_tool_text() -> str:
    return "标题：某篇论文\n内容：" + "很长的检索原文" * 500


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens([]) == 0

    def test_cjk_counts_more_than_ascii(self):
        cjk = [HumanMessage(content="中" * 100)]
        ascii_msg = [HumanMessage(content="a" * 100)]
        assert estimate_tokens(cjk) > estimate_tokens(ascii_msg)


class TestWindowMessages:
    def test_under_threshold_passthrough(self):
        messages = _turn("问题", "短的检索结果", "回答")
        trimmed = window_messages(messages, max_tokens=10000)
        assert trimmed == messages

    def test_over_threshold_old_tool_message_summarized(self):
        messages = (
            _turn("第一轮问题", _big_tool_text(), "第一轮回答")
            + _turn("第二轮问题", "短结果", "第二轮回答")
        )
        trimmed = window_messages(messages, max_tokens=150)

        # 第一轮的工具消息被摘要化（大幅缩短）。
        old_tool = next(m for m in trimmed if isinstance(m, ToolMessage) and m.tool_call_id == "call_第一轮问题")
        assert len(str(old_tool.content)) < 200
        assert "第一轮问题" in old_tool.tool_call_id

    def test_current_turn_always_intact(self):
        messages = (
            _turn("旧轮问题", _big_tool_text(), "旧轮回答")
            + _turn("本轮问题", _big_tool_text(), "本轮回答")
        )
        trimmed = window_messages(messages, max_tokens=150)

        # 本轮（最后一条用户消息起）一个字不动。
        last_human_index = max(i for i, m in enumerate(trimmed) if isinstance(m, HumanMessage))
        assert trimmed[last_human_index:] == messages[len(messages) - len(trimmed[last_human_index:]):]
        assert any(
            isinstance(m, ToolMessage) and len(str(m.content)) > 1000
            for m in trimmed[last_human_index:]
        )

    def test_tool_call_pairing_preserved(self):
        messages = _turn("问题", _big_tool_text(), "回答") * 5
        trimmed = window_messages(messages, max_tokens=200)

        tool_call_ids = {
            call["id"]
            for message in trimmed
            if isinstance(message, AIMessage)
            for call in (getattr(message, "tool_calls", None) or [])
        }
        tool_message_ids = {
            m.tool_call_id for m in trimmed if isinstance(m, ToolMessage)
        }
        assert tool_call_ids == tool_message_ids  # 一一对应，不多不少

    def test_deterministic(self):
        messages = _turn("问题", _big_tool_text(), "回答") * 4
        first = window_messages(messages, max_tokens=300)
        second = window_messages(messages, max_tokens=300)
        assert first == second  # 同一输入同一输出 → 压完定死


class TestWindowHook:
    def _hook(self, make_settings, **overrides):
        settings = make_settings(**overrides)
        return context_module.build_window_hook(settings)

    def test_passthrough_returns_empty_dict(self, make_settings):
        hook = self._hook(make_settings, context_window_max_tokens=100000)
        state = {"messages": _turn("问题", "短结果", "回答")}
        assert hook(state) == {}

    def test_over_threshold_returns_llm_input_messages(self, make_settings):
        hook = self._hook(make_settings, context_window_max_tokens=100)
        state = {"messages": _turn("问题", _big_tool_text(), "回答") * 5}
        result = hook(state)
        assert "llm_input_messages" in result
        assert len(result["llm_input_messages"]) < len(state["messages"]) or all(
            len(str(getattr(m, "content", ""))) < 500
            for m in result["llm_input_messages"][:4]
        )

    def test_dispatch_prefers_window_over_slim(self, make_settings):
        settings = make_settings(
            context_window_enabled=True,
            context_slim_enabled=True,
        )
        hook = build_context_hook(settings)
        state = {"messages": _turn("问题", "短结果", "回答")}
        # 窗口模式下阈值内透传返回空 dict；slim 模式会返回精简消息。
        assert hook(state) == {}

    def test_dispatch_falls_back_to_slim(self, make_settings):
        settings = make_settings(
            context_window_enabled=False,
            context_slim_enabled=True,
        )
        hook = build_context_hook(settings)
        state = {"messages": _turn("问题", _big_tool_text(), "回答") * 3}
        result = hook(state)
        assert "llm_input_messages" in result

    def test_dispatch_returns_none_when_all_disabled(self, make_settings):
        settings = make_settings(context_window_enabled=False, context_slim_enabled=False)
        assert build_context_hook(settings) is None
