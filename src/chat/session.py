"""会话管理：内存版上下文存储 + token 统计（后续可换 Redis）。

Author: MADENG
Reviewer: Li Rongdong
"""
from __future__ import annotations

import uuid
from collections import defaultdict, deque


# 按 session_id 保存近 N 轮对话上下文，并统计 token 用量。
#
# Attributes:
#     _max_rounds: 保留的最大对话轮数（每轮对应 user + assistant 两条消息）。
#     _store: session_id -> deque[消息字典]，自动滚动到最近 N*2 条。
#     _token_stats: session_id -> 累计 token 统计。
#
# Notes:
#     后续可换 Redis：仅需把 ``_store`` / ``_token_stats`` 替换为 Redis 客户端调用。
class SessionManager:
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

    # 创建新会话。
    #
    # Returns:
    #     新生成的 session_id。
    def create(self) -> str:
        sid = uuid.uuid4().hex
        self._store[sid] = deque(maxlen=self._max_rounds * 2)
        self._token_stats[sid] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        return sid

    # 获取会话历史（按时间顺序）。
    #
    # Args:
    #     session_id: 会话 id。
    #
    # Returns:
    #     消息字典列表；会话不存在时返回空列表。
    def get_history(self, session_id: str) -> list[dict[str, str]]:
        return list(self._store.get(session_id, []))

    # 追加一轮对话。
    #
    # Args:
    #     session_id: 会话 id。
    #     role: 角色（``"user"`` / ``"assistant"``）。
    #     content: 消息内容。
    def append(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._store:
            self._store[session_id] = deque(maxlen=self._max_rounds * 2)
        self._store[session_id].append({"role": role, "content": content})

    # 清空会话（含 token 统计）。
    #
    # Args:
    #     session_id: 会话 id。
    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)
        self._token_stats.pop(session_id, None)

    # 累计 token 用量。
    #
    # Args:
    #     session_id: 会话 id。
    #     prompt: 本轮 prompt tokens。
    #     completion: 本轮 completion tokens。
    def add_tokens(self, session_id: str, prompt: int = 0, completion: int = 0) -> None:
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

    # 获取当前会话累计 token 统计。
    #
    # Args:
    #     session_id: 会话 id。
    #
    # Returns:
    #     累计 token 字典（含 ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``）。
    def get_tokens(self, session_id: str) -> dict[str, int]:
        return dict(
            self._token_stats.get(
                session_id,
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )
        )
