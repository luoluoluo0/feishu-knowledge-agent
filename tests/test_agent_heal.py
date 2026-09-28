"""断连自愈的定向测试：悬空尾段检测与自愈函数。

消息对象用 SimpleNamespace 模拟（只用到 type/tool_calls/tool_call_id/id），
不建真实 Agent。
"""

from types import SimpleNamespace

from app.agent import find_dangling_tail, heal_dangling_tool_history


def human():
    return SimpleNamespace(type="human", id="h")


def ai(calls=None, id="a"):
    return SimpleNamespace(type="ai", id=id, content="", tool_calls=calls or [])


def tool_call(call_id):
    return {"id": call_id, "name": "search_paper", "args": {}, "type": "tool_call"}


def tool_msg(call_id, tid="t"):
    return SimpleNamespace(type="tool", id=tid, name="search_paper",
                           tool_call_id=call_id, content="资料1")


class TestFindDanglingTail:
    def test_dangling_last_ai_detected(self):
        msgs = [human(), ai([tool_call("c1")], id="a-dangling")]
        dangling = find_dangling_tail(msgs)
        assert [m.id for m in dangling] == ["a-dangling"]

    def test_healthy_history_returns_empty(self):
        msgs = [human(), ai([tool_call("c1")]), tool_msg("c1"), ai()]
        assert find_dangling_tail(msgs) == []

    def test_complete_rounds_return_empty(self):
        msgs = [
            human(),
            ai([tool_call("c1")]),
            tool_msg("c1"),
            ai([tool_call("c2")]),
            tool_msg("c2"),
            ai(),
        ]
        assert find_dangling_tail(msgs) == []

    def test_partial_answers_whole_tail_removed(self):
        """AI 同时要了 c1/c2，只有 c1 有结果——整段按悬空处理。"""
        msgs = [
            human(),
            ai([tool_call("c1"), tool_call("c2")]),
            tool_msg("c1"),
        ]
        dangling = find_dangling_tail(msgs)
        assert len(dangling) == 2  # AI + 那条已答的 ToolMessage 一起删

    def test_new_human_message_is_boundary(self):
        """悬空段之后用户又发了新消息？不可能——新问题会先撞校验。
        但悬空段前有历史 human 边界时不能误删更早的轮次。"""
        msgs = [
            human(),
            ai([tool_call("c1")]),
            tool_msg("c1"),
            ai(),  # 第一轮正常结束
            human(),  # 第二轮提问
            ai([tool_call("c9")]),  # 断连留下的悬空
        ]
        dangling = find_dangling_tail(msgs)
        assert [m.id for m in dangling] == ["a"]  # 只删最后那条（id 默认 "a"）


def test_heal_removes_dangling_and_returns_count():
    dangling_msg = ai([tool_call("c9")], id="a-dangling")
    captured = {}

    def get_state(config):
        return SimpleNamespace(values={"messages": [human(), dangling_msg]})

    def update_state(config, update):
        captured["update"] = update

    fake_agent = SimpleNamespace(get_state=get_state, update_state=update_state)
    removed = heal_dangling_tool_history(fake_agent, {})

    assert removed == 1
    removals = captured["update"]["messages"]
    assert len(removals) == 1
    assert removals[0].id == "a-dangling"


def test_heal_noop_on_healthy_history():
    fake_agent = SimpleNamespace(
        get_state=lambda config: SimpleNamespace(
            values={"messages": [human(), ai([tool_call("c1")]), tool_msg("c1"), ai()]}
        ),
        update_state=lambda config, update: (_ for _ in ()).throw(
            AssertionError("健康历史不该触发 update_state")
        ),
    )
    assert heal_dangling_tool_history(fake_agent, {}) == 0


def test_heal_swallows_state_errors():
    def boom(config):
        raise RuntimeError("状态读不了")

    assert heal_dangling_tool_history(SimpleNamespace(get_state=boom), {}) == 0
