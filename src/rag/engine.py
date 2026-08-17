"""RAG 引擎：检索 -> 生成（含无 LLM 降级）的顶层编排。

统一返回结构：``{"answer": str, "sources": list[str]}``。
LLM 生成模式与降级模式都会返回命中的知识来源。

流式能力：:func:`stream_chat` 返回生成器，逐块产出 answer 片段，
并在结束时产出 token 用量与 sources。

本模块是 RAG 流程的「编排层」，依赖关系：
    engine -> pipeline(context/prompts) + retriever -> core(...)
不包含检索/分片/向量化等具体实现，仅负责串联调用。

Author: MADENG
Reviewer: Li Rongdong
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from langchain_core.output_parsers import StrOutputParser

from src.chat.intent import classify
from src.config import Settings
from src.rag.llm import build_chat_model
from src.rag.pipeline.context import (
    build_source_objs,
    format_docs,
    format_history,
)
from src.rag.pipeline.prompts import build_prompt
from src.rag.retriever import build_retriever

logger = logging.getLogger(__name__)


# 构建 RAG 链。
#
# Args:
#     settings: 全局配置。
#
# Returns:
#     - 配置了 LLM Key：返回「检索 -> Prompt -> LLM -> 文本」链。
#     - 未配置 LLM Key：返回「检索 -> 片段拼接」降级链。
#
# Notes:
#     返回的链可被当作 ``chain(payload)`` 调用，
#     其中 ``payload = {"question": str, "history": list[dict]}``。
def build_rag_chain(settings: Settings) -> Any:
    retriever = build_retriever(settings)

    if settings.llm.available:
        llm = build_chat_model(settings.llm)
        prompt = build_prompt(settings.llm.system_role)

        # LCEL 链：检索 + 生成，同时保留 sources 以便返回
        def _run_with_sources(x: dict[str, Any]) -> dict[str, Any]:
            docs = retriever.invoke(x["question"])
            context = format_docs(docs)  # 未命中时为 "(无相关知识片段)"
            history = format_history(x.get("history", []))
            answer = prompt | llm | StrOutputParser() | (lambda a: a)
            result = answer.invoke(
                {"context": context, "history": history, "question": x["question"]}
            )
            return {"answer": result, "sources": build_source_objs(docs)}

        logger.info(
            "RAG 链已构建（LLM 生成模式，provider=%s, system_role=%s）",
            settings.llm.provider,
            settings.llm.system_role,
        )
        return _run_with_sources

    # 降级模式：返回命中片段原文 + 出处，不调用 LLM
    def _fallback(x: dict[str, Any]) -> dict[str, Any]:
        docs = retriever.invoke(x["question"])
        if not docs:
            return {
                "answer": "当前知识库中未找到相关信息。（未配置 LLM，无法生成友好回复）",
                "sources": [],
            }
        answer = "\n\n".join(
            f"[{i}] {d.metadata.get('source', '未知')}\n{d.page_content}"
            for i, d in enumerate(docs, start=1)
        )
        return {
            "answer": "（未配置 LLM，以下为命中的知识片段原文）\n\n" + answer,
            "sources": build_source_objs(docs),
        }

    logger.info("RAG 链已构建（降级模式：返回检索片段原文）")
    return _fallback


# 构建供 /chat 接口调用的处理器，统一返回 dict。
#
# Args:
#     settings: 全局配置。
#
# Returns:
#     可调用对象 ``handler(question, history) -> dict``，
#     返回 ``{"answer": str, "sources": list}``。
def build_chat_handler(settings: Settings) -> Any:
    chain = build_rag_chain(settings)

    def _invoke(question: str, history: list[dict[str, str]]) -> dict[str, Any]:
        payload = {"question": question, "history": history}
        # LLM 模式与降级模式均返回 dict
        result = chain(payload)
        return result if isinstance(result, dict) else {"answer": result, "sources": []}

    return _invoke


# 流式问答：逐块产出生成内容，结束产出元信息。
#
# Args:
#     settings: 全局配置。
#     question: 用户当前提问。
#     history: 历史对话消息列表。
#
# Yields:
#     事件 dict 结构：
#       - ``{"type": "sources", "sources": [...]}`` 检索命中来源
#       - ``{"type": "delta", "content": "..."}``    生成片段
#       - ``{"type": "done", "tokens": {...}}``      结束，含 token 用量
#
# Notes:
#     未配置 LLM 或未命中时，也会产出 done 事件（此时 delta 即完整 answer）。
def stream_chat(
    settings: Settings,
    question: str,
    history: list[dict[str, str]],
) -> Iterator[dict[str, Any]]:
    retriever = build_retriever(settings)

    # 闲聊类问题直接走预设友好回复，不检索（避免冷冰冰的「未找到」）
    intent, greeting_reply = classify(question)
    if intent == "chat" and greeting_reply:
        yield {"type": "sources", "sources": []}
        yield {"type": "delta", "content": greeting_reply}
        yield {
            "type": "done",
            "sources": [],
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return

    # 检索
    docs = retriever.invoke(question)
    # 普通用户问答：不回传原文明文（with_content=False），仅暴露来源文件名 + 相关度
    source_objs = build_source_objs(docs, with_content=False)

    yield {"type": "sources", "sources": source_objs}

    # 降级模式：直接返回原文，不调用 LLM
    if not settings.llm.available:
        if not docs:
            yield {
                "type": "delta",
                "content": "当前知识库中未找到相关信息。（未配置 LLM，无法生成友好回复）",
            }
            yield {
                "type": "done",
                "sources": [],
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            return
        answer = "\n\n".join(
            f"[{i}] {d.metadata.get('source', '未知')}\n{d.page_content}"
            for i, d in enumerate(docs, start=1)
        )
        content = "（未配置 LLM，以下为命中的知识片段原文）\n\n" + answer
        yield {"type": "delta", "content": content}
        yield {
            "type": "done",
            "sources": source_objs,
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return

    # LLM 流式生成（无论是否命中都过一遍 LLM，让知识库助手角色自然回复）
    llm = build_chat_model(settings.llm, streaming=True)
    prompt = build_prompt(settings.llm.system_role)
    context = format_docs(docs)  # 未命中时为 "(无相关知识片段)"
    history_str = format_history(history)

    prompt_tokens = 0
    completion_tokens = 0
    for chunk in llm.stream(
        prompt.format_messages(context=context, history=history_str, question=question)
    ):
        # token 用量在 usage_metadata（input_tokens / output_tokens），
        # 兼容旧字段名（prompt_tokens / completion_tokens）
        usage = getattr(chunk, "usage_metadata", None) or {}
        if usage:
            prompt_tokens = usage.get("input_tokens", usage.get("prompt_tokens", prompt_tokens))
            completion_tokens = usage.get(
                "output_tokens", usage.get("completion_tokens", completion_tokens)
            )
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        if content:
            yield {"type": "delta", "content": content}

    yield {
        "type": "done",
        "sources": source_objs,
        "tokens": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
