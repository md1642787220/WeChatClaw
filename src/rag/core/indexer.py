"""索引构建模块：编排「分片 -> 向量化 -> 持久化」完成建库。

本模块是 RAG 流程的「入库编排层」，将分片模块、向量化模块、持久化模块
串起来，对上层提供一个简洁的建库/入库入口，屏蔽底层细节。

依赖方向：indexer -> (splitter, embedder, store)，单向无环。

Author: MADENG
Reviewer: Li Rongdong
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.rag.core.embedder import Embeddings
from src.rag.core.splitter import split_markdown
from src.rag.core.store import add_documents, build_index


# 将文本切分为 Document 列表（不做向量化）。
#
# Args:
#     text: 文档全文。
#     source: 来源文件标识，写入元数据。
#
# Returns:
#     ``Document`` 列表，每个元素对应一个文本块。
#
# Notes:
#     该函数仅做分片，是「分片 -> 入库」流程的中间产物，
#     可供预览（仅解析不入库）场景直接复用。
def split_text_to_documents(text: str, source: str) -> list[Document]:
    chunks = split_markdown(text, source=source)
    return [
        Document(page_content=c["content"], metadata=c["metadata"]) for c in chunks
    ]


# 从文本列表全量构建向量索引。
#
# Args:
#     texts: 文本列表。
#     sources: 与 texts 等长的来源标识列表。
#     embeddings: Embedding 实例。
#     collection_name: 集合名。
#     persist_dir: 持久化目录。
#
# Returns:
#     Chroma 向量库实例。
#
# Notes:
#     全量重建会覆盖既有集合，增量场景请使用 ``index_documents``。
def build_from_texts(
    texts: list[str],
    sources: list[str],
    embeddings: Embeddings,
    collection_name: str,
    persist_dir: str,
) -> Chroma:
    documents: list[Document] = []
    for text, source in zip(texts, sources):
        documents.extend(split_text_to_documents(text, source))
    return build_index(documents, embeddings, collection_name, persist_dir)


# 增量入库：将已切分的 Document 列表写入向量库（按稳定 ID 去重）。
#
# Args:
#     documents: 已切分的文档块。
#     embeddings: Embedding 实例。
#     collection_name: 集合名。
#     persist_dir: 持久化目录。
#
# Returns:
#     ``(新增块数, 跳过重复块数)`` 二元组。
def index_documents(
    documents: list[Document],
    embeddings: Embeddings,
    collection_name: str,
    persist_dir: str,
) -> tuple[int, int]:
    return add_documents(documents, embeddings, collection_name, persist_dir)
