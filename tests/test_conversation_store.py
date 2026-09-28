from app.conversation_store import (
    append_turn,
    clear_history,
    get_db_path,
    load_history,
)
from app.coreference import Turn


def make_turn(question: str, item_id: str = "001", answer: str = "回答") -> Turn:
    return Turn(
        question=question,
        rewritten_query=f"改写后的：{question}",
        item_id=item_id,
        answer=answer,
    )


def db_settings(make_settings, tmp_path):
    return make_settings(conversation_db_path=str(tmp_path / "history.db"))


def test_load_history_is_empty_when_db_missing(make_settings, tmp_path):
    settings = db_settings(make_settings, tmp_path)

    assert load_history("thread-a", settings=settings) == []


def test_append_then_load_roundtrip(make_settings, tmp_path):
    settings = db_settings(make_settings, tmp_path)

    append_turn("thread-a", make_turn("第一篇讲了什么？"), settings=settings)
    history = load_history("thread-a", settings=settings)

    assert len(history) == 1
    assert history[0].question == "第一篇讲了什么？"
    assert history[0].rewritten_query == "改写后的：第一篇讲了什么？"
    assert history[0].item_id == "001"
    assert history[0].answer == "回答"


def test_load_history_returns_most_recent_in_ascending_order(make_settings, tmp_path):
    settings = db_settings(make_settings, tmp_path)

    for index in range(10):
        append_turn("thread-a", make_turn(f"问题{index}"), settings=settings)

    history = load_history("thread-a", limit=5, settings=settings)

    assert len(history) == 5
    assert [turn.question for turn in history] == [
        "问题5",
        "问题6",
        "问题7",
        "问题8",
        "问题9",
    ]


def test_answer_is_truncated_to_500_chars(make_settings, tmp_path):
    settings = db_settings(make_settings, tmp_path)

    append_turn("thread-a", make_turn("问题", answer="字" * 900), settings=settings)
    history = load_history("thread-a", settings=settings)

    assert len(history[0].answer) == 500


def test_different_threads_do_not_mix(make_settings, tmp_path):
    settings = db_settings(make_settings, tmp_path)

    append_turn("thread-a", make_turn("A 的问题"), settings=settings)
    append_turn("thread-b", make_turn("B 的问题"), settings=settings)

    assert [t.question for t in load_history("thread-a", settings=settings)] == [
        "A 的问题"
    ]
    assert [t.question for t in load_history("thread-b", settings=settings)] == [
        "B 的问题"
    ]


def test_clear_history_removes_only_target_thread(make_settings, tmp_path):
    settings = db_settings(make_settings, tmp_path)

    append_turn("thread-a", make_turn("A1"), settings=settings)
    append_turn("thread-a", make_turn("A2"), settings=settings)
    append_turn("thread-b", make_turn("B1"), settings=settings)

    deleted = clear_history("thread-a", settings=settings)

    assert deleted == 2
    assert load_history("thread-a", settings=settings) == []
    assert len(load_history("thread-b", settings=settings)) == 1


def test_get_db_path_uses_settings_value(make_settings, tmp_path):
    settings = db_settings(make_settings, tmp_path)

    assert get_db_path(settings).name == "history.db"
