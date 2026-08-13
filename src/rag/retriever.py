"""检索器构建：从 Chroma 向量库生成 retriever。"""
from __future__ import annotations

from typing import Any

from src.config import Settings
from src.knowledge.embeddings import build_embeddings
from src.knowledge.vector_store import load_store, store_exists


def build_retriever(settings: Settings) -> Any:
    """构建检索器。

    - 向量库已构建：从持久化目录加载。
    - 尚未构建：返回一个空结果占位检索器（提示先执行索引构建脚本）。
    """
    embeddings = build_embeddings(
        model_name=settings.embedding.model,
        device=settings.embedding.device,
    )

    if store_exists(settings.vector_db.persist_dir, settings.vector_db.collection):
        store = load_store(
            embeddings=embeddings,
            collection_name=settings.vector_db.collection,
            persist_dir=settings.vector_db.persist_dir,
        )
        return store.as_retriever(search_kwargs={"k": settings.retrieval.top_k})

    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    class _EmptyRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str) -> list[Document]:
            return []

    return _EmptyRetriever()
