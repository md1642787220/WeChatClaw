"""测试：会话管理。"""
from src.chat.session import SessionManager


def test_create_returns_id():
    mgr = SessionManager(max_rounds=10)
    sid = mgr.create()
    assert sid
    assert mgr.get_history(sid) == []


def test_append_and_get_history():
    mgr = SessionManager(max_rounds=10)
    sid = mgr.create()
    mgr.append(sid, "user", "你好")
    mgr.append(sid, "assistant", "你好，有什么可以帮您？")
    history = mgr.get_history(sid)
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "你好"}


def test_history_limited_by_rounds():
    mgr = SessionManager(max_rounds=2)
    sid = mgr.create()
    for i in range(5):
        mgr.append(sid, "user", f"q{i}")
        mgr.append(sid, "assistant", f"a{i}")
    history = mgr.get_history(sid)
    # 最多保留 2 轮 = 4 条消息
    assert len(history) == 4
    assert history[0]["content"] == "q3"


def test_clear():
    mgr = SessionManager(max_rounds=10)
    sid = mgr.create()
    mgr.append(sid, "user", "hello")
    mgr.clear(sid)
    assert mgr.get_history(sid) == []


def test_append_unknown_session_auto_create():
    mgr = SessionManager(max_rounds=10)
    mgr.append("nonexistent", "user", "hello")
    assert mgr.get_history("nonexistent") == [{"role": "user", "content": "hello"}]


def test_token_stats_default_zero():
    mgr = SessionManager(max_rounds=10)
    sid = mgr.create()
    assert mgr.get_tokens(sid) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_token_stats_accumulate():
    mgr = SessionManager(max_rounds=10)
    sid = mgr.create()
    mgr.add_tokens(sid, prompt=100, completion=50)
    mgr.add_tokens(sid, prompt=30, completion=20)
    stats = mgr.get_tokens(sid)
    assert stats["prompt_tokens"] == 130
    assert stats["completion_tokens"] == 70
    assert stats["total_tokens"] == 200


def test_token_stats_cleared_with_session():
    mgr = SessionManager(max_rounds=10)
    sid = mgr.create()
    mgr.add_tokens(sid, prompt=10, completion=5)
    mgr.clear(sid)
    assert mgr.get_tokens(sid)["total_tokens"] == 0
