from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import api as api_module
from app.auth import require_api_key, require_current_user
from app.users import AuthUser
from app.coreference import RewriteResult
from app.planner import PlanResult
from app.rate_limit import require_rate_limit


@pytest.fixture
def client():
    """绕过鉴权与限流依赖，只验证响应体结构。"""

    api_module.app.dependency_overrides[require_api_key] = lambda: True
    api_module.app.dependency_overrides[require_rate_limit] = lambda: None
    # 问答/清理端点已切换到用户依赖：以 service 遗留身份通过，
    # 归属校验对 user_id=0 不设限（管理通道语义）。
    api_module.app.dependency_overrides[require_current_user] = lambda: AuthUser(
        user_id=0, username="service", via="api_key"
    )
    try:
        yield TestClient(api_module.app)
    finally:
        api_module.app.dependency_overrides.clear()


def stub_rewrite(
    question: str, rewritten_query: str = "", item_id: str = ""
) -> RewriteResult:
    return RewriteResult(
        original_question=question,
        rewritten_query=rewritten_query or question,
        item_id=item_id,
        rewritten=bool(rewritten_query),
        reason="测试用。",
    )


def test_chat_response_exposes_rewrite_fields(monkeypatch, client):
    monkeypatch.setattr(
        api_module,
        "run_agent",
        lambda agent, question, thread_id: {
            "messages": [
                SimpleNamespace(content="这是回答。", tool_calls=[], type="ai")
            ],
            "rewrite": stub_rewrite(
                question, "第001篇文献的研究方法是什么？", "001"
            ),
            "intent": SimpleNamespace(
                question="第001篇文献的研究方法是什么？",
                intent="summary_paper",
                reason="测试用。",
            ),
        },
    )
    monkeypatch.setattr(api_module, "get_tool_agent", lambda: SimpleNamespace())
    monkeypatch.setattr(api_module, "record_agent_log", lambda **_: "log-1")
    monkeypatch.setattr(
        api_module,
        "build_trace_summary",
        lambda messages, answer="": {"tool_calls": [], "sources": []},
    )

    response = client.post(
        "/chat",
        json={"question": "它的研究方法是什么？", "thread_id": "t1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["original_question"] == "它的研究方法是什么？"
    assert body["rewritten_query"] == "第001篇文献的研究方法是什么？"
    assert body["resolved_item_id"] == "001"
    assert body["rewrite_reason"] == "测试用。"
    assert body["intent"] == "summary_paper"
    assert body["intent_reason"] == "测试用。"


def test_planner_chat_response_exposes_rewrite_and_intent(monkeypatch, client):
    monkeypatch.setattr(
        api_module,
        "get_planner_agent",
        lambda: SimpleNamespace(
            answer=lambda question, thread_id: {
                "question": question,
                "answer": "这是回答。",
                "plan": PlanResult(task_type="simple_qa", steps=[], reason="测试"),
                "step_results": [],
                "rewrite": stub_rewrite(
                    question, "第001篇文献的汇报提纲", "001"
                ),
                "intent": SimpleNamespace(
                    question=question,
                    intent="group_report",
                    reason="测试用。",
                ),
            }
        ),
    )
    monkeypatch.setattr(api_module, "record_agent_log", lambda **_: "log-2")

    response = client.post(
        "/planner-chat", json={"question": "给我一份提纲", "thread_id": "t1"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["original_question"] == "给我一份提纲"
    assert body["rewritten_query"] == "第001篇文献的汇报提纲"
    assert body["resolved_item_id"] == "001"
    assert body["intent"] == "group_report"
    assert body["intent_reason"] == "测试用。"


def test_clear_thread_response_exposes_deleted_turns(monkeypatch, client):
    monkeypatch.setattr(api_module, "clear_cached_agent", lambda mode: None)
    monkeypatch.setattr(
        api_module,
        "clear_thread_checkpoints",
        lambda thread_id: {"thread_id": thread_id, "deleted_checkpoints": 1},
    )
    monkeypatch.setattr(api_module, "clear_history", lambda thread_id: 3)

    response = client.post("/admin/clear-thread", json={"thread_id": "t1"})

    assert response.status_code == 200
    assert response.json()["deleted_turns"] == 3


def stub_prepared(question: str, thread_id: str, intent: str):
    """构造 prepare_question 的返回值，只带 api.py 会用到的字段。"""

    return SimpleNamespace(
        thread_id=thread_id,
        original_question=question,
        rewrite=stub_rewrite(question),
        intent=SimpleNamespace(question=question, intent=intent, reason="测试用。"),
        config={},
        settings=SimpleNamespace(),
    )


def test_ask_routes_group_report_to_planner(monkeypatch, client):
    monkeypatch.setattr(
        api_module,
        "prepare_question",
        lambda question, thread_id, mode: stub_prepared(
            question, thread_id, "group_report"
        ),
    )
    monkeypatch.setattr(
        api_module,
        "get_planner_agent",
        lambda: SimpleNamespace(
            answer=lambda question, thread_id, prepared: {
                "question": question,
                "answer": "提纲如下。",
                "plan": PlanResult(task_type="group_report", steps=[], reason="测试"),
                "step_results": [],
                "rewrite": prepared.rewrite,
                "intent": prepared.intent,
            }
        ),
    )
    monkeypatch.setattr(api_module, "record_agent_log", lambda **_: "log-3")

    response = client.post("/ask", json={"question": "给我一份提纲", "thread_id": "t1"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "planner_agent"
    assert body["intent"] == "group_report"
    assert body["answer"] == "提纲如下。"
    assert "plan_text" in body


def test_ask_routes_metadata_query_to_tool_agent(monkeypatch, client):
    monkeypatch.setattr(
        api_module,
        "prepare_question",
        lambda question, thread_id, mode: stub_prepared(
            question, thread_id, "metadata_query"
        ),
    )
    monkeypatch.setattr(
        api_module,
        "run_agent",
        lambda agent, question, thread_id, prepared: {
            "messages": [
                SimpleNamespace(content="缺失清单如下。", tool_calls=[], type="ai")
            ],
            "rewrite": prepared.rewrite,
            "intent": prepared.intent,
        },
    )
    monkeypatch.setattr(api_module, "get_tool_agent", lambda: SimpleNamespace())
    monkeypatch.setattr(api_module, "record_agent_log", lambda **_: "log-4")
    monkeypatch.setattr(
        api_module,
        "build_trace_summary",
        lambda messages, answer="": {"tool_calls": [], "sources": []},
    )

    response = client.post("/ask", json={"question": "哪些文献缺 PDF", "thread_id": "t1"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "tool_agent"
    assert body["intent"] == "metadata_query"
    assert body["answer"] == "缺失清单如下。"


def test_ask_prepares_the_question_only_once(monkeypatch, client):
    """统一入口必须先预处理再路由，两条链路共用同一份结果。"""

    calls = []

    def fake_prepare(question, thread_id, mode):
        calls.append((question, thread_id, mode))
        return stub_prepared(question, thread_id, "simple_qa")

    monkeypatch.setattr(api_module, "prepare_question", fake_prepare)
    monkeypatch.setattr(
        api_module,
        "run_agent",
        lambda agent, question, thread_id, prepared: {
            "messages": [SimpleNamespace(content="回答。", tool_calls=[], type="ai")],
            "rewrite": prepared.rewrite,
            "intent": prepared.intent,
        },
    )
    monkeypatch.setattr(api_module, "get_tool_agent", lambda: SimpleNamespace())
    monkeypatch.setattr(api_module, "record_agent_log", lambda **_: "log-5")
    monkeypatch.setattr(
        api_module,
        "build_trace_summary",
        lambda messages, answer="": {"tool_calls": [], "sources": []},
    )

    response = client.post("/ask", json={"question": "什么是生计韧性", "thread_id": "t1"})

    assert response.status_code == 200
    assert calls == [("什么是生计韧性", "t1", "auto")]
