"""上下文格式化模块：将检索到的文档组织为 Prompt 上下文与来源列表。

本模块从引擎中剥离出「检索结果 -> 文本/结构」的纯格式化逻辑，
供生成阶段与知识库管理接口复用，避免重复实现。

核心约定：
- 按 ``source`` 文件名对文档去重，同一文件多个块合并为一个引用。
- 引用编号从 1 开始连续编号，与 Prompt 中的 [1][2] 一一对应。

Author: MADENG
Reviewer: Li Rongdong
"""

from __future__ import annotations

from typing import Any

# 未命中时写入上下文的占位文本，便于 LLM 识别「无命中」分支
_NO_CONTEXT_HINT = "（无相关知识片段）"

# 来源元数据缺失时的兜底标识
_UNKNOWN_SOURCE = "未知来源"


# 按 source 文件名去重，保留首次出现的顺序。
#
# Args:
#     docs: 检索到的文档列表（每个含 ``metadata["source"]``）。
#
# Returns:
#     ``[(index, source, chunks), ...]``，index 从 1 开始连续编号，
#     同一文件的多个 chunk 合并到同一引用下。
def dedupe_by_source(docs: list[Any]) -> list[tuple[int, str, list[Any]]]:
    seen: dict[str, int] = {}  # source -> 列表索引
    result: list[tuple[int, str, list[Any]]] = []
    for doc in docs:
        src = doc.metadata.get("source") or _UNKNOWN_SOURCE
        if src in seen:
            result[seen[src]][2].append(doc)
        else:
            seen[src] = len(result)
            result.append((len(result) + 1, src, [doc]))
    return result


# 将检索到的文档块拼接为带编号的上下文文本。
#
# Args:
#     docs: 检索到的文档列表。
#
# Returns:
#     按 source 去重后的编号上下文；空列表返回「无相关知识片段」提示。
#
# Notes:
#     编号与 ``build_source_objs`` 中的 index 对应。
def format_docs(docs: list[Any]) -> str:
    if not docs:
        return _NO_CONTEXT_HINT
    parts: list[str] = []
    for idx, source, chunks in dedupe_by_source(docs):
        body = "\n---\n".join(c.page_content for c in chunks)
        parts.append(f"[{idx}] {source}\n{body}")
    return "\n\n".join(parts)


# 构造结构化来源列表（按 source 去重）。
#
# Args:
#     docs: 检索到的文档列表。
#     with_content: True 时包含片段原文（管理员预览用）；False 时仅返回
#         ``index/source/score/chunk_count``，不泄露原文明文（普通用户问答用）。
#
# Returns:
#     ``[{"index", "source", "score", "chunk_count", "content"?}, ...]``。
#     编号 index 与 ``format_docs`` 中的 [1][2] 对应。
def build_source_objs(docs: list[Any], with_content: bool = True) -> list[dict[str, Any]]:
    objs: list[dict[str, Any]] = []
    for idx, source, chunks in dedupe_by_source(docs):
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


# 格式化历史对话为文本。
#
# Args:
#     history: 历史消息列表，每项 ``{"role": ..., "content": ...}``。
#
# Returns:
#     每行一条 ``role: content`` 的文本；空历史返回「（无）」。
def format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "（无）"
    lines = [f"{m['role']}: {m['content']}" for m in history]
    return "\n".join(lines)
