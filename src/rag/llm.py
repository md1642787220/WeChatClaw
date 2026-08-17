"""LLM 构建：基于 OpenAI 兼容协议（DeepSeek 等），支持流式与 token 统计。

Author: MADENG
Reviewer: Li Rongdong
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.config import LLMConfig


# 构建 ChatModel。DeepSeek 等走 OpenAI 兼容协议。
#
# Args:
#     config: LLM 配置。
#     streaming: 是否开启流式输出。开启后可用 ``.stream()`` 逐块获取。
#
# Returns:
#     实现 LangChain ``BaseChatModel`` 接口的 ChatModel 实例。
def build_chat_model(config: LLMConfig, streaming: bool = False) -> BaseChatModel:
    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        timeout=config.timeout_seconds,
        streaming=streaming,
    )
