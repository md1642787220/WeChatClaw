"""LLM 构建：基于 OpenAI 兼容协议（DeepSeek 等）。"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.config import LLMConfig


def build_chat_model(config: LLMConfig) -> BaseChatModel:
    """构建 ChatModel。DeepSeek 等走 OpenAI 兼容协议。"""
    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        timeout=config.timeout_seconds,
    )
