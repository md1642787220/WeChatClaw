"""检索器构建：从 Chroma 向量库生成带相似度阈值过滤的检索器。"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config import Settings
from src.knowledge.embeddings import build_embeddings
from src.knowledge.vector_store import load_store, store_exists

logger = logging.getLogger(__name__)


class _EmptyRetriever(BaseRetriever):
    """空检索器：向量库未构建时使用，返回空结果。"""

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return []


class ThresholdRetriever(BaseRetriever):
    """基于相似度阈值过滤的检索器。

    Chroma 返回的距离（distance）越小表示越相似，本项目向量已归一化，
    使用余弦相似度，其值域为 [0, 1]，越大越相似。
    """

    store: Any
    top_k: int
    threshold: float

    def _get_relevant_documents(self, query: str) -> list[Document]:
        results = self.store.similarity_search_with_relevance_scores(
            query, k=self.top_k
        )
        docs: list[Document] = []
        for doc, score in results:
            if score < self.threshold:
                continue
            # 将相似度写入 metadata，便于后续溯源/展示
            doc.metadata["score"] = round(score, 4)
            docs.append(doc)
        return docs


def build_retriever(settings: Settings) -> BaseRetriever:
    """构建检索器。

    - 向量库已构建：加载 Chroma，返回带阈值过滤的检索器。
    - 尚未构建：返回空占位检索器（不加载 embedding 模型，避免启动时下载）。
    """
    if not store_exists(settings.vector_db.persist_dir, settings.vector_db.collection):
        logger.info("向量库尚未构建，使用空检索器（请先执行入库）")
        return _EmptyRetriever()

    embeddings = build_embeddings(
        model_name=settings.embedding.model,
        device=settings.embedding.device,
    )
    store = load_store(
        embeddings=embeddings,
        collection_name=settings.vector_db.collection,
        persist_dir=settings.vector_db.persist_dir,
    )
    return ThresholdRetriever(
        store=store,
        top_k=settings.retrieval.top_k,
        threshold=settings.retrieval.threshold,
    )
