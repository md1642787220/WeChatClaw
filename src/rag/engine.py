"""RAG 引擎：检索 + Prompt + 生成（含无 LLM 降级）。

统一返回结构：{"answer": str, "sources": list[str]}。
LLM 生成模式与降级模式都会返回命中的知识来源。

流式能力：stream_chat() 返回生成器，逐块产出 answer 片段，
并在结束时产出 token 用量与 sources。
"""
from __future__ import annotations

import logging
from typing import Any, Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.chat.intent import classify
from src.config import Settings
from src.rag.llm import build_chat_model
from src.rag.retriever import build_retriever

logger = logging.getLogger(__name__)

# system 模板：角色名由 settings.llm.system_role 注入（默认"知识库助手"）。
# 遵循 OpenAI 兼容 messages 规范：messages[0].role="system"。
_SYSTEM_TEMPLATE = """你是企业内部{role}（角色：{role}）。请仅依据【知识片段】回答员工问题，
回答需准确、简洁，语气专业且友好。

引用要求（重要）：每个知识片段前已带编号 [1]、[2]……请在回答中**对应事实所在的句末**
使用方括号标注来源编号，例如“公司年假为 10 天[1]”。可在同一句后并列多个编号，如[1][3]。
若多个片段都支持同一句话，可写 [1][2]。禁止编造来源编号。

无命中处理：当下方【知识片段】为空或不足以回答时（如当前问题与知识库无关，或只是
闲聊/自我介绍/能力询问），请**仍以{role}身份**礼貌回复：
  - 可以简要介绍你能做什么（基于企业内部知识库回答员工关于制度、流程、FAQ 等问题）；
  - 引导用户换一种更具体的问题；
  - 不要编造公司制度/数据；不要硬说"未找到"，除非用户问的就是具体业务问题；
  - 不要泄露内部敏感信息。"""

# human 消息：携带知识片段、历史与当前问题
_HUMAN_TEMPLATE = """【知识片段】
{context}

【历史对话】
{history}

【员工问题】
{question}
"""


def _build_prompt(system_role: str) -> ChatPromptTemplate:
    """根据 system_role 构造 ChatPromptTemplate。

    遵循 OpenAI 兼容 messages 规范：messages[0].role="system"，messages[1].role="user"（human）。
    """
    system_msg = _SYSTEM_TEMPLATE.format(role=system_role)
    return ChatPromptTemplate.from_messages(
        [("system", system_msg), ("human", _HUMAN_TEMPLATE)]
    )


def _dedupe_by_source(docs: list[Any]) -> list[tuple[int, str, list[Any]]]:
    """按 source 文件名去重，保留首次出现的顺序。

    同一文件被切出多个 chunk 时，合并到同一引用下。返回 [(index, source, chunks), ...]。
    index 从 1 开始连续编号，与 _format_docs / _build_source_objs 中的 [1][2] 一一对应。
    """
    seen: dict[str, int] = {}     # source -> 列表索引
    result: list[tuple[int, str, list[Any]]] = []
    for doc in docs:
        src = doc.metadata.get("source") or "未知来源"
        if src in seen:
            result[seen[src]][2].append(doc)
        else:
            seen[src] = len(result)
            result.append((len(result) + 1, src, [doc]))
    return result


def _format_docs(docs: list[Any]) -> str:
    """将检索到的文档块拼接为带编号的上下文（按 source 去重，编号与来源列表一一对应）。

    空列表返回明确提示，便于 LLM 识别"无命中"情况并走自我介绍分支。
    """
    if not docs:
        return "（无相关知识片段）"
    parts: list[str] = []
    for idx, source, chunks in _dedupe_by_source(docs):
        body = "\n---\n".join(c.page_content for c in chunks)
        parts.append(f"[{idx}] {source}\n{body}")
    return "\n\n".join(parts)


def _build_source_objs(docs: list[Any], with_content: bool = True) -> list[dict[str, Any]]:
    """构造结构化来源列表（按 source 去重）。

    返回: [{"index": 1, "source": "文件名", "score": 0.x, "chunk_count": n, "content": ...}, ...]
    编号 index 与 _format_docs 中的 [1][2] 对应。同一文件的多个 chunk 合并为单个引用。

    :param with_content: True 时包含片段原文（管理员预览用）；False 时仅返回
        ``index/source/score/chunk_count``，不泄露原文明文（普通用户问答用）。
    """
    objs: list[dict[str, Any]] = []
    for idx, source, chunks in _dedupe_by_source(docs):
        # 取该 source 中分数最高（最小距离）者作为代表分数
        best_score: Any = None
        for c in chunks:
            s = c.metadata.get("score")
            if s is None and hasattr(c, "metadata"):
                s = c.metadata.get("distance")
            if s is None:
                continue
            if best_score is None or s < best_score:
                best_score = s
        obj: dict[str, Any] = {
            "index": idx,
            "source": source,
            "score": best_score,
            "chunk_count": len(chunks),
        }
        if with_content:
            obj["content"] = "\n---\n".join(c.page_content for c in chunks)
        objs.append(obj)
    return objs


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
        prompt = _build_prompt(settings.llm.system_role)

        # LCEL 链：检索 + 生成，同时保留 sources 以便返回
        def _run_with_sources(x: dict[str, Any]) -> dict[str, Any]:
            docs = retriever.invoke(x["question"])
            context = _format_docs(docs)  # 未命中时为 "(无相关知识片段)"
            history = _format_history(x.get("history", []))
            answer = prompt | llm | StrOutputParser() | (lambda a: a)
            result = answer.invoke(
                {"context": context, "history": history, "question": x["question"]}
            )
            return {"answer": result, "sources": _build_source_objs(docs)}

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
            "sources": _build_source_objs(docs),
        }

    logger.info("RAG 链已构建（降级模式：返回检索片段原文）")
    return _fallback


def build_chat_handler(settings: Settings) -> Any:
    """构建供 /chat 接口调用的处理器，统一返回 dict（含 answer 与 sources）。"""
    chain = build_rag_chain(settings)

    def _invoke(question: str, history: list[dict[str, str]]) -> dict[str, Any]:
        payload = {"question": question, "history": history}
        # LLM 模式与降级模式均返回 dict
        result = chain(payload)
        return result if isinstance(result, dict) else {"answer": result, "sources": []}

    return _invoke


def stream_chat(
    settings: Settings,
    question: str,
    history: list[dict[str, str]],
) -> Iterator[dict[str, Any]]:
    """流式问答：逐块产出生成内容，结束产出元信息。

    产出的事件 dict 结构：
      - {"type": "sources", "sources": [...]}  检索命中来源
      - {"type": "delta", "content": "..."}    生成片段
      - {"type": "done", "tokens": {...}}      结束，含 token 用量

    未配置 LLM 或未命中时，也会产出 done 事件（此时 delta 即完整 answer）。
    """
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
    source_objs = _build_source_objs(docs, with_content=False)

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
    prompt = _build_prompt(settings.llm.system_role)
    context = _format_docs(docs)  # 未命中时为 "(无相关知识片段)"
    history_str = _format_history(history)

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
