"""持久化模块：向量库（Chroma）的加载、写入与存在性判断。

本模块封装对向量库持久化的所有读写操作，是 RAG 流程的「落地层」。
上层（索引构建、检索器）只通过本模块提供的函数访问向量库，
不直接操作 Chroma 对象，保持职责单一。

设计要点：
- 用稳定 ID（来源 + 内容哈希）去重，重复入库自动跳过。
- 维护一个全局索引版本号，供上层感知向量库变更（如缓存失效）。
- 指定 ``hnsw:space: cosine``，与已 L2 归一化的向量匹配，
  避免默认 L2 距离导致相似度分数异常。

Author: MADENG
Reviewer: Li Rongdong
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# 向量库变更信号：每次入库时递增，供上层判断是否需要重建检索器/缓存
_index_version = 0

# 集合元数据：指定余弦相似度空间。
# 本项目向量已做 L2 归一化（见 embedder），余弦相似度值域 [0,1]，越大越相似，
# 与 Chroma 的 similarity_search_with_relevance_scores 语义一致。
# 注意：若不指定，Chroma 默认使用 L2 距离（越小越相似，且可 >1），
# 与归一化向量不匹配，会导致相似度分数异常。
_COLLECTION_METADATA = {"hnsw:space": "cosine"}


# 返回当前向量库版本号。
#
# Returns:
#     整数版本号，每次成功入库递增。
def get_index_version() -> int:
    return _index_version


# 向量库内容变更时调用，递增版本号。
#
# Notes:
#     该函数仅修改进程内全局变量，进程重启后归零，
#     用于在单次进程生命周期内触发缓存失效。
def bump_index_version() -> None:
    global _index_version
    _index_version += 1


# 基于来源 + 内容生成稳定 ID，重复入库时自动去重。
#
# Args:
#     doc: 文档块。
#
# Returns:
#     MD5 十六进制字符串，作为向量库中的唯一 ID。
#
# Notes:
#     ID 仅依赖 ``source`` 元数据与正文，与向量无关，
#     保证同一内容多次入库结果一致。
def _doc_id(doc: Document) -> str:
    raw = f"{doc.metadata.get('source', '')}\n{doc.page_content}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# 构建向量索引并持久化（全量重建）。
#
# Args:
#     docs: 已切分的文档块。
#     embeddings: Embedding 实例。
#     collection_name: 集合名。
#     persist_dir: 持久化目录。
#
# Returns:
#     Chroma 向量库实例。
#
# Notes:
#     该方法会覆盖目标集合的既有内容，适用于离线全量建库场景；
#     增量场景请使用 ``add_documents``。
def build_index(
    docs: list[Document],
    embeddings: Embeddings,
    collection_name: str,
    persist_dir: str,
) -> Chroma:
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        collection_metadata=_COLLECTION_METADATA,
        persist_directory=persist_dir,
    )


# 加载已持久化的向量库（只读打开）。
#
# Args:
#     embeddings: 与建库时一致的 Embedding 实例（维度/模型须匹配）。
#     collection_name: 集合名。
#     persist_dir: 持久化目录。
#
# Returns:
#     Chroma 向量库实例。
#
# Notes:
#     调用前应先通过 ``store_exists`` 确认库已构建，
#     否则 Chroma 会创建一个空库而非报错。
def load_store(
    embeddings: Embeddings,
    collection_name: str,
    persist_dir: str,
) -> Chroma:
    return Chroma(
        embedding_function=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )


# 判断向量库是否已构建。
#
# Args:
#     persist_dir: 持久化目录。
#     collection_name: 集合名（保留参数，供扩展使用，当前未参与判断）。
#
# Returns:
#     True 表示向量库已存在。
#
# Notes:
#     Chroma 以 ``chroma.sqlite3`` 元数据库文件为标志，
#     collection 数据存于 UUID 子目录，故仅需判断该文件是否存在。
def store_exists(persist_dir: str, collection_name: str | None = None) -> bool:
    base = Path(persist_dir)
    if not base.exists():
        return False
    return (base / "chroma.sqlite3").exists()


# 增量入库：新增文档块，按稳定 ID 去重。
#
# Args:
#     docs: 待入库的文档块。
#     embeddings: Embedding 实例。
#     collection_name: 集合名。
#     persist_dir: 持久化目录。
#
# Returns:
#     ``(新增块数, 跳过重复块数)`` 二元组。
#
# Notes:
#     - 已存在集合则加载后追加，否则新建。
#     - 只有实际新增时才触发索引版本号递增。
def add_documents(
    docs: list[Document],
    embeddings: Embeddings,
    collection_name: str,
    persist_dir: str,
) -> tuple[int, int]:
    # 已存在的集合则加载，否则创建
    if store_exists(persist_dir):
        store = load_store(embeddings, collection_name, persist_dir)
        existing_ids = set(store.get()["ids"])
    else:
        store = Chroma(
            embedding_function=embeddings,
            collection_name=collection_name,
            collection_metadata=_COLLECTION_METADATA,
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
