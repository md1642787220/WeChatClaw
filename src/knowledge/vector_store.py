"""向量库封装：Chroma 持久化存储与索引构建。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# 向量库变更信号：每次入库时递增，供上层判断是否需要重建检索器/缓存
_index_version = 0


def get_index_version() -> int:
    """返回当前向量库版本号。"""
    return _index_version


def bump_index_version() -> None:
    """向量库内容变更时调用，递增版本号。"""
    global _index_version
    _index_version += 1


def _doc_id(doc: Document) -> str:
    """基于来源 + 内容生成稳定 ID，重复入库时自动去重。"""
    raw = f"{doc.metadata.get('source', '')}\n{doc.page_content}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


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


def add_documents(
    docs: list[Document],
    embeddings: Embeddings,
    collection_name: str,
    persist_dir: str,
) -> tuple[int, int]:
    """增量入库：新增文档块，按稳定 ID 去重。

    Args:
        docs: 待入库的文档块。
        embeddings: Embedding 实例。
        collection_name: 集合名。
        persist_dir: 持久化目录。

    Returns:
        (新增块数, 跳过重复块数)。
    """
    # 已存在的集合则加载，否则创建
    if store_exists(persist_dir):
        store = load_store(embeddings, collection_name, persist_dir)
        existing_ids = set(store.get()["ids"])
    else:
        store = Chroma(
            embedding_function=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )
        existing_ids: set[str] = set()

    new_docs: list[Document] = []
    new_ids: list[str] = []
    skipped = 0
    for doc in docs:
        did = _doc_id(doc)
        if did in existing_ids:
            skipped += 1
            continue
        new_docs.append(doc)
        new_ids.append(did)
        existing_ids.add(did)

    if new_docs:
        store.add_documents(documents=new_docs, ids=new_ids)
        bump_index_version()

    return len(new_docs), skipped
