"""会话管理：内存版上下文存储（后续可换 Redis）。"""
from __future__ import annotations

import uuid
from collections import defaultdict, deque


class SessionManager:
    """按 session_id 保存近 N 轮对话上下文。"""

    def __init__(self, max_rounds: int = 10) -> None:
        self._max_rounds = max_rounds
        # session_id -> deque[{"role": "user"|"assistant", "content": str}]
        self._store: dict[str, deque[dict[str, str]]] = defaultdict(
            lambda: deque(maxlen=max_rounds * 2)
        )

    def create(self) -> str:
        """创建新会话，返回 session_id。"""
        sid = uuid.uuid4().hex
        self._store[sid] = deque(maxlen=self._max_rounds * 2)
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
        """清空会话。"""
        self._store.pop(session_id, None)
