"""向量库封装：Chroma 持久化存储与索引构建。"""
from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def build_index(
    docs: list[Document],
    embeddings: Embeddings,
    collection_name: str,
    persist_dir: str,
) -> Chroma:
    """构建向量索引并持久化。

    Args:
        docs: 已切分的文档块。
        embeddings: Embedding 实例。
        collection_name: 集合名。
        persist_dir: 持久化目录。

    Returns:
        Chroma 向量库实例。
    """
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )


def load_store(
    embeddings: Embeddings,
    collection_name: str,
    persist_dir: str,
) -> Chroma:
    """加载已持久化的向量库。"""
    return Chroma(
        embedding_function=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )


def store_exists(persist_dir: str, collection_name: str | None = None) -> bool:
    """判断向量库是否已构建。

    Chroma 持久化目录以 chroma.sqlite3 元数据库为标志，
    collection 数据存于 UUID 子目录中，故仅需判断 sqlite 文件是否存在。
    """
    base = Path(persist_dir)
    if not base.exists():
        return False
    return (base / "chroma.sqlite3").exists()
