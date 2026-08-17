"""检索器模块：从向量库生成带相似度阈值过滤的检索器。

本模块是 RAG 流程的「检索入口」，组合 core 层的向量化、持久化、排序模块，
对上层提供一个 ``BaseRetriever`` 接口。

提供模块级缓存：启动时通过 :func:`warmup_retriever` 预加载 embedding 模型
与向量库，后续请求复用同一实例，避免每次请求都重新加载模型。

Author: MADENG
Reviewer: Li Rongdong
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config import Settings
from src.rag.core.embedder import build_embeddings
from src.rag.core.reranker import filter_by_threshold
from src.rag.core.store import load_store, store_exists

logger = logging.getLogger(__name__)

# 模块级缓存：检索器单例，避免重复加载 embedding 模型
_retriever_cache: dict[str, BaseRetriever] = {}


# 空检索器：向量库未构建时使用，返回空结果。
#
# Notes:
#     该类不加载 embedding 模型，避免服务启动时因库不存在而触发下载。
class _EmptyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> list[Document]:
        return []


# 基于相似度阈值过滤的检索器。
#
# Attributes:
#     store: Chroma 向量库实例。
#     top_k: 检索返回的最大文档数。
#     threshold: 相似度阈值（余弦相似度，值域 [0, 1]）。
#
# Notes:
#     内部先调用向量库做相似度检索，再用 reranker 模块按阈值过滤。
#     过滤后的分数会写入 ``doc.metadata["score"]``，供上游溯源展示。
class ThresholdRetriever(BaseRetriever):
    store: Any
    top_k: int
    threshold: float

    def _get_relevant_documents(self, query: str) -> list[Document]:
        results = self.store.similarity_search_with_relevance_scores(
            query, k=self.top_k
        )
        docs = [doc for doc, _ in results]
        scores = [score for _, score in results]
        # 交由排序模块做阈值过滤（分数同步写入 metadata["score"]）
        pairs = filter_by_threshold(docs, scores, self.threshold)
        return [doc for doc, _ in pairs]


# 根据配置关键字段生成缓存键。
#
# Args:
#     settings: 全局配置。
#
# Returns:
#     由向量库路径、集合名、模型名/设备、检索参数拼接的字符串。
#
# Notes:
#     任一关键字段变化都会产生不同键，从而触发检索器重建。
def _cache_key(settings: Settings) -> str:
    return "|".join(
        [
            settings.vector_db.persist_dir,
            settings.vector_db.collection,
            settings.embedding.model,
            settings.embedding.device,
            str(settings.retrieval.top_k),
            str(settings.retrieval.threshold),
        ]
    )


# 构建检索器（带缓存）。
#
# Args:
#     settings: 全局配置。
#     use_cache: True 时复用缓存的检索器，避免重复加载模型。
#
# Returns:
#     - 向量库已构建：返回带阈值过滤的检索器。
#     - 尚未构建：返回空占位检索器（不加载 embedding 模型）。
#
# Notes:
#     首次构建会加载 embedding 模型，成本较高（秒级），
#     建议通过 :func:`warmup_retriever` 在启动阶段预热。
def build_retriever(settings: Settings, use_cache: bool = True) -> BaseRetriever:
    if not store_exists(settings.vector_db.persist_dir, settings.vector_db.collection):
        logger.info("向量库尚未构建，使用空检索器（请先执行入库）")
        return _EmptyRetriever()

    key = _cache_key(settings)
    if use_cache and key in _retriever_cache:
        return _retriever_cache[key]

    embeddings = build_embeddings(
        model_name=settings.embedding.model,
        device=settings.embedding.device,
    )
    store = load_store(
        embeddings=embeddings,
        collection_name=settings.vector_db.collection,
        persist_dir=settings.vector_db.persist_dir,
    )
    retriever = ThresholdRetriever(
        store=store,
        top_k=settings.retrieval.top_k,
        threshold=settings.retrieval.threshold,
    )
    if use_cache:
        _retriever_cache[key] = retriever
    return retriever


# 启动预热：预加载 embedding 模型与向量库，填充缓存。
#
# Args:
#     settings: 全局配置。
#
# Notes:
#     在 FastAPI lifespan 中调用，使首个 /chat、/kb/search 请求不再
#     承担模型加载开销，实现「启动完成后知识库即就绪」。
#     向量库未构建时直接跳过，降级为首次检索时懒加载。
def warmup_retriever(settings: Settings) -> None:
    if not store_exists(settings.vector_db.persist_dir, settings.vector_db.collection):
        logger.info("向量库尚未构建，跳过预热（可在管理后台入库后重启或首次检索时自动加载）")
        return
    logger.info(
        "预热知识库：加载 embedding 模型与向量库（persist_dir=%s, collection=%s）...",
        settings.vector_db.persist_dir,
        settings.vector_db.collection,
    )
    build_retriever(settings, use_cache=True)
    logger.info("知识库预热完成，后续请求将直接复用缓存的检索器。")


# 向量库内容变更（入库）后调用，清除缓存以便下次重建。
#
# Notes:
#     调用后下一个检索请求会重新加载向量库与 embedding，
#     从而反映最新入库的数据。
def invalidate_retriever_cache() -> None:
    _retriever_cache.clear()
