"""LLM 构建：用 OpenAI 兼容协议（DeepSeek 等）造一个聊天模型，支持流式。

Author: MADENG
Reviewer: Li Rongdong
"""
from langchain_openai import ChatOpenAI

from src.config import LLMConfig


# 造一个聊天模型。DeepSeek 这些都走 OpenAI 兼容协议。
#
# 参数：
#     llm_config: LLM 配置。
#     streaming: 要不要开流式输出。开了之后能用 .stream() 一块一块地拿。
#
# 返回：
#     一个能用的聊天模型。
def make_chat_model(llm_config: LLMConfig, streaming=False):
    return ChatOpenAI(
        model=llm_config.model,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        temperature=llm_config.temperature,
        timeout=llm_config.timeout_seconds,
        streaming=streaming,
    )
