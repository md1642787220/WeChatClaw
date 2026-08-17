"""排序模块：对检索结果做相似度过滤与排序。

本模块是 RAG 流程中「检索后、生成前」的一环，负责：
- 按相似度阈值过滤低相关结果；
- 对命中结果按相似度排序；
- 将相似度分数写入每个文档的 metadata，便于溯源与展示。

注意：本模块只做「排序/过滤」这类纯逻辑操作，不依赖向量库或模型，
保持与其它模块解耦，方便后续替换更复杂的重排策略（如 cross-encoder）。

Author: MADENG
Reviewer: Li Rongdong
"""

from __future__ import annotations

from langchain_core.documents import Document

# 相似度分数写入 metadata 的键名，供上游溯源/展示统一读取
SCORE_KEY = "score"


# 按相似度阈值过滤检索结果，并写入分数元数据。
#
# Args:
#     docs: 检索到的文档列表（与 scores 等长）。
#     scores: 与 docs 一一对应的相似度分数。
#     threshold: 相似度阈值，低于该值的文档被丢弃。
#
# Returns:
#     ``(Document, score)`` 列表，仅包含 >= threshold 的命中项，
#     分数已四舍五入到 4 位小数并写入 ``doc.metadata["score"]``。
#
# Notes:
#     不会修改 ``docs`` / ``scores`` 输入本身，
#     而是返回新列表；但会对 doc 的 metadata 做原地写入（便于下游直接读取）。
def filter_by_threshold(
    docs: list[Document],
    scores: list[float],
    threshold: float,
) -> list[tuple[Document, float]]:
    result: list[tuple[Document, float]] = []
    for doc, score in zip(docs, scores):
        if score < threshold:
            continue
        doc.metadata[SCORE_KEY] = round(score, 4)
        result.append((doc, score))
    return result


# 按相似度分数对检索结果排序。
#
# Args:
#     pairs: ``(Document, score)`` 列表。
#     reverse: True 表示分数从高到低（默认）；False 表示从低到高。
#
# Returns:
#     排序后的 Document 列表（不含分数）。
#
# Notes:
#     排序为稳定排序，分数相同时保持输入相对顺序。
def sort_by_score(
    pairs: list[tuple[Document, float]], reverse: bool = True
) -> list[Document]:
    pairs = sorted(pairs, key=lambda p: p[1], reverse=reverse)
    return [doc for doc, _ in pairs]


# 对检索结果执行「阈值过滤 + 排序」的组合操作。
#
# Args:
#     docs: 检索到的文档列表。
#     scores: 对应相似度分数。
#     threshold: 相似度阈值。
#
# Returns:
#     过滤后按相似度降序排列的 Document 列表。
#
# Notes:
#     该函数等价于 ``sort_by_score(filter_by_threshold(...))``，
#     是检索器常用的便捷入口。
def rerank(
    docs: list[Document], scores: list[float], threshold: float
) -> list[Document]:
    pairs = filter_by_threshold(docs, scores, threshold)
    return sort_by_score(pairs)
