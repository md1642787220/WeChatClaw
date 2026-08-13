"""RAG 引擎：LCEL 链编排，检索 + Prompt + 生成（含无 LLM 降级）。"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import Settings
from src.rag.llm import build_chat_model
from src.rag.retriever import build_retriever

logger = logging.getLogger(__name__)

_SYSTEM_TEMPLATE = """你是企业内部知识库助手。请仅依据【知识片段】回答员工问题，
回答需准确、简洁，并注明依据的知识来源（片段标题或文件名）。
若片段不足以回答，请明确说明"当前知识库中未找到相关信息"，禁止编造，也不要泄露内部敏感信息。

【知识片段】
{context}

【历史对话】
{history}

【员工问题】
{question}
"""

_PROMPT = ChatPromptTemplate.from_template(_SYSTEM_TEMPLATE)


def _format_docs(docs: list[Any]) -> str:
    """将检索到的文档块拼接为上下文。"""
    parts: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"[来源: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def _format_history(history: list[dict[str, str]]) -> str:
    """格式化历史对话。"""
    if not history:
        return "（无）"
    lines = [f"{m['role']}: {m['content']}" for m in history]
    return "\n".join(lines)


def build_rag_chain(settings: Settings) -> Any:
    """构建 RAG 链。

    - 配置了 LLM Key：返回「检索 → Prompt → LLM → 文本」链。
    - 未配置 LLM Key：返回「检索 → 片段拼接」降级链。
    """
    retriever = build_retriever(settings)

    if settings.llm.available:
        llm = build_chat_model(settings.llm)
        chain = (
            {
                "context": retriever | _format_docs,
                "history": lambda x: _format_history(x.get("history", [])),
                "question": lambda x: x["question"],
            }
            | _PROMPT
            | llm
            | StrOutputParser()
        )
        logger.info("RAG 链已构建（LLM 生成模式，provider=%s）", settings.llm.provider)
        return chain

    # 降级模式：返回命中片段原文 + 出处，不调用 LLM
    def _fallback(x: dict[str, Any]) -> dict[str, Any]:
        docs = retriever.invoke(x["question"])
        if not docs:
            return {"answer": "当前知识库中未找到相关信息。", "sources": []}
        answer = "\n\n".join(
            f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}" for d in docs
        )
        return {
            "answer": "（未配置 LLM，以下为命中的知识片段原文）\n\n" + answer,
            "sources": [d.metadata.get("source", "") for d in docs],
        }

    logger.info("RAG 链已构建（降级模式：返回检索片段原文）")
    return _fallback


def build_chat_handler(settings: Settings) -> Any:
    """构建供 /chat 接口调用的处理器，统一返回 dict（含 answer 与 sources）。"""
    chain = build_rag_chain(settings)

    def _invoke(question: str, history: list[dict[str, str]]) -> dict[str, Any]:
        payload = {"question": question, "history": history}
        # LCEL Runnable 用 invoke；降级模式 chain 是普通函数，直接调用
        result = chain.invoke(payload) if hasattr(chain, "invoke") else chain(payload)

        if isinstance(result, dict):
            return result
        return {"answer": result, "sources": []}

    return _invoke
