from types import SimpleNamespace

import pytest

from app import observability
from app.token_meter import TokenMeter


@pytest.fixture(autouse=True)
def reset_observability_state(monkeypatch):
    monkeypatch.setattr(observability, "_langfuse_client", None)
    monkeypatch.setattr(observability, "_langfuse_config_key", None)


def test_disabled_tracing_builds_config_without_callback(make_settings):
    config = observability.build_run_config(
        thread_id="thread-1",
        mode="tool-agent",
        settings=make_settings(),
    )

    # TokenMeter 恒挂（token 计量不依赖 Langfuse 开关）；关 trace 时
    # callbacks 里应该只有它，没有 Langfuse handler。
    callbacks = config["callbacks"]
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], TokenMeter)
    assert config["configurable"]["token_meter"] is callbacks[0]
    assert config["configurable"]["thread_id"] == "thread-1"
    assert config["metadata"]["langfuse_session_id"] == "thread-1"
    assert config["metadata"]["langfuse_tags"] == [
        "feishu-paper-agent",
        "tool-agent",
    ]


def test_enabled_tracing_rejects_missing_credentials(make_settings):
    settings = make_settings(langfuse_tracing_enabled=True)

    with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY"):
        observability.initialize_observability(settings)


def test_enabled_tracing_initializes_once_and_adds_callback(monkeypatch, make_settings):
    created_clients = []
    created_handlers = []

    class FakeLangfuse:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created_clients.append(self)

        def shutdown(self):
            return None

    def fake_handler(**kwargs):
        handler = SimpleNamespace(kwargs=kwargs)
        created_handlers.append(handler)
        return handler

    monkeypatch.setattr(observability, "Langfuse", FakeLangfuse)
    monkeypatch.setattr(observability, "CallbackHandler", fake_handler)

    settings = make_settings(
        langfuse_tracing_enabled=True,
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key="sk-lf-test",
    )
    first_config = observability.build_run_config(
        thread_id="thread-2",
        mode="planner-agent",
        settings=settings,
    )
    second_config = observability.build_run_config(
        thread_id="thread-2",
        mode="planner-agent",
        settings=settings,
    )

    assert len(created_clients) == 1
    assert len(created_handlers) == 2
    assert created_clients[0].kwargs["public_key"] == "pk-lf-test"
    # callbacks = [TokenMeter, Langfuse handler]：计量器在前，trace 在后。
    assert isinstance(first_config["callbacks"][0], TokenMeter)
    assert first_config["callbacks"][1] is created_handlers[0]
    assert second_config["callbacks"][1] is created_handlers[1]
    assert first_config["configurable"]["token_meter"] is first_config["callbacks"][0]


def test_shutdown_is_safe_and_resets_client(monkeypatch):
    calls = []
    fake_client = SimpleNamespace(shutdown=lambda: calls.append("shutdown"))
    monkeypatch.setattr(observability, "_langfuse_client", fake_client)
    monkeypatch.setattr(
        observability,
        "_langfuse_config_key",
        ("public", "secret", "url", "test"),
    )

    observability.shutdown_observability()

    assert calls == ["shutdown"]
    assert observability._langfuse_client is None
    assert observability._langfuse_config_key is None
