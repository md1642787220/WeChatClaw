"""会话管理：内存版上下文存储 + token 统计（后续可换 Redis）。"""
from __future__ import annotations

import uuid
from collections import defaultdict, deque


class SessionManager:
    """按 session_id 保存近 N 轮对话上下文，并统计 token 用量。"""

    def __init__(self, max_rounds: int = 10) -> None:
        self._max_rounds = max_rounds
        # session_id -> deque[{"role": "user"|"assistant", "content": str}]
        self._store: dict[str, deque[dict[str, str]]] = defaultdict(
            lambda: deque(maxlen=max_rounds * 2)
        )
        # session_id -> 累计 token 统计
        self._token_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )

    def create(self) -> str:
        """创建新会话，返回 session_id。"""
        sid = uuid.uuid4().hex
        self._store[sid] = deque(maxlen=self._max_rounds * 2)
        self._token_stats[sid] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        return sid

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """获取会话历史（按时间顺序）。"""
        return list(self._store.get(session_id, []))

    def append(self, session_id: str, role: str, content: str) -> None:
        """追加一轮对话。"""
        if session_id not in self._store:
            self._store[session_id] = deque(maxlen=self._max_rounds * 2)
        self._store[session_id].append({"role": role, "content": content})

    def clear(self, session_id: str) -> None:
        """清空会话（含 token 统计）。"""
        self._store.pop(session_id, None)
        self._token_stats.pop(session_id, None)

    def add_tokens(self, session_id: str, prompt: int = 0, completion: int = 0) -> None:
        """累计 token 用量。"""
        if session_id not in self._token_stats:
            self._token_stats[session_id] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        stats = self._token_stats[session_id]
        stats["prompt_tokens"] += prompt
        stats["completion_tokens"] += completion
        stats["total_tokens"] += prompt + completion

    def get_tokens(self, session_id: str) -> dict[str, int]:
        """获取当前会话累计 token 统计。"""
        return dict(
            self._token_stats.get(
                session_id,
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )
        )
