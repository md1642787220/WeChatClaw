"""会话管理：用内存存上下文和 token 统计（以后可换成 Redis）。

Author: MADENG
Reviewer: Li Rongdong
"""
import uuid
from collections import defaultdict, deque


# 按 session_id 保存最近 N 轮对话，并统计 token 用量。
#
# 属性：
#     max_rounds: 最多保留几轮对话（每轮对应 user + assistant 两条消息）。
#     _store: session_id -> 消息队列，自动只留最近 N*2 条。
#     _token_stats: session_id -> 累计 token 统计。
#
# 注意：
#     以后换 Redis：只要把 _store / _token_stats 换成 Redis 客户端就行。
class SessionManager:
    def __init__(self, max_rounds=10):
        self.max_rounds = max_rounds
        # session_id -> 消息队列（{"role": "user"/"assistant", "content": 文字}）
        self._store = defaultdict(self._make_message_deque)
        # session_id -> 累计 token 统计
        self._token_stats = defaultdict(self._make_empty_token_stats)

    # 内部：造一个长度受限的队列（上限 = max_rounds * 2 条消息）。
    def _make_message_deque(self):
        return deque(maxlen=self.max_rounds * 2)

    # 内部：造一份全是 0 的 token 统计。
    def _make_empty_token_stats(self):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # 开一个新会话。
    #
    # 返回：
    #     新生成的 session_id。
    def create(self):
        session_id = uuid.uuid4().hex
        self._store[session_id] = self._make_message_deque()
        self._token_stats[session_id] = self._make_empty_token_stats()
        return session_id

    # 拿到会话历史（按时间顺序）。
    #
    # 参数：
    #     session_id: 会话 id。
    #
    # 返回：
    #     消息字典列表；会话不存在时返回空列表。
    def get_history(self, session_id):
        message_deque = self._store.get(session_id)
        if message_deque is None:
            return []
        return list(message_deque)

    # 追加一轮对话。
    #
    # 参数：
    #     session_id: 会话 id。
    #     role: 角色（"user" 或 "assistant"）。
    #     content: 消息内容。
    def append(self, session_id, role, content):
        if session_id not in self._store:
            self._store[session_id] = self._make_message_deque()
        message = {"role": role, "content": content}
        self._store[session_id].append(message)

    # 清空会话（连 token 统计一起）。
    def clear(self, session_id):
        if session_id in self._store:
            del self._store[session_id]
        if session_id in self._token_stats:
            del self._token_stats[session_id]

    # 累计 token 用量。
    #
    # 参数：
    #     session_id: 会话 id。
    #     prompt: 本轮 prompt tokens。
    #     completion: 本轮 completion tokens。
    def add_tokens(self, session_id, prompt=0, completion=0):
        if session_id not in self._token_stats:
            self._token_stats[session_id] = self._make_empty_token_stats()
        stats = self._token_stats[session_id]
        stats["prompt_tokens"] = stats["prompt_tokens"] + prompt
        stats["completion_tokens"] = stats["completion_tokens"] + completion
        stats["total_tokens"] = stats["prompt_tokens"] + stats["completion_tokens"]

    # 拿到当前会话累计的 token 统计。
    #
    # 返回：
    #     累计 token 字典（含 prompt_tokens / completion_tokens / total_tokens）。
    def get_tokens(self, session_id):
        stats = self._token_stats.get(session_id)
        if stats is None:
            return self._make_empty_token_stats()
        return dict(stats)
