"""持久化模块：负责把向量库（Chroma）读出来、写进去、判断建没建好。

这个模块管所有跟「存向量」有关的操作，是 RAG 流程的「落地的最后一步」。
上层（建库、检索）都通过这里的函数来碰向量库，不直接操作 Chroma 对象，
大家各管一摊，不乱。

几个设计要点：
- 用「来源 + 内容」算一个固定 ID，重复入库的会自动跳过。
- 维护一个全局的库版本号，让上层知道库变了（比如好清缓存）。
- 指定用余弦相似度（hnsw:space: cosine），跟已经归一化的向量对得上，
  避免默认的 L2 距离把相似度分数算得乱七八糟。

Author: MADENG
Reviewer: Li Rongdong
"""
import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# 库版本号：每入库一次就加一，上层靠它判断要不要重建检索器/缓存
_index_version = 0

# 集合元数据：告诉 Chroma 用余弦相似度。
# 我们项目里的向量已经做过归一化（见 embedder），余弦相似度的范围是 0 到 1，
# 越大越像，跟 Chroma 的 similarity_search_with_relevance_scores 的意思一致。
# 注意：不指定的话 Chroma 默认用 L2 距离（越小越像，还可能大于 1），
# 跟归一化向量对不上，分数会算错。
_collection_metadata = {"hnsw:space": "cosine"}


# 返回当前向量库的版本号。
def get_index_version():
    return _index_version


# 向量库内容变了就调这个，把版本号加一。
#
# 注意：
#     它只改进程里的全局变量，进程重启后就归零了，
#     作用就是在一个进程的存活期间提醒上层「库变了，快清缓存」。
def bump_index_version():
    global _index_version
    _index_version = _index_version + 1


# 根据「来源 + 内容」算一个固定 ID，重复入库时自动去重。
#
# 参数：
#     one_doc: 一个文档块。
#
# 返回：
#     一串 MD5 十六进制文字，作为向量库里的唯一 ID。
#
# 注意：
#     ID 只跟来源和正文有关，跟向量无关，所以同一内容不管入库几次，ID 都一样。
def _make_doc_id(one_doc: Document):
    source_value = one_doc.metadata.get("source", "")
    raw_text = str(source_value) + "\n" + one_doc.page_content
    md5 = hashlib.md5()
    md5.update(raw_text.encode("utf-8"))
    return md5.hexdigest()


# 全量建库：把文档块向量化后写进向量库，会覆盖旧的。
#
# 参数：
#     doc_list: 切好的文档块。
#     embeddings: 向量化工具。
#     collection_name: 集合名。
#     persist_dir: 存到哪个目录。
#
# 返回：
#     Chroma 向量库实例。
#
# 注意：
#     这个方法会覆盖目标集合里原来的内容，适合离线一次性建库；
#     想增量加的话请用「增量入库」。
def build_full_index(doc_list, embeddings: Embeddings, collection_name: str, persist_dir: str):
    return Chroma.from_documents(
        documents=doc_list,
        embedding=embeddings,
        collection_name=collection_name,
        collection_metadata=_collection_metadata,
        persist_directory=persist_dir,
    )


# 打开已经存好的向量库（只读打开）。
#
# 参数：
#     embeddings: 跟建库时一致的向量化工具（模型和维度必须对得上）。
#     collection_name: 集合名。
#     persist_dir: 存向量库的目录。
#
# 返回：
#     Chroma 向量库实例。
#
# 注意：
#     调它之前应该先用「向量库存在吗」确认库建好了，
#     不然 Chroma 会新建一个空库，而不是报错。
def open_vector_store(embeddings: Embeddings, collection_name: str, persist_dir: str):
    return Chroma(
        embedding_function=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )


# 判断向量库建没建好。
#
# 参数：
#     persist_dir: 存向量库的目录。
#     collection_name: 集合名（先留着以后扩展用，现在没参与判断）。
#
# 返回：
#     建好了返回 True，没建返回 False。
#
# 注意：
#     Chroma 用 chroma.sqlite3 这个元数据库文件当标志，
#     集合数据存在 UUID 子目录里，所以只要判断这个文件在不在就行。
def vector_store_exists(persist_dir: str, collection_name=None):
    base_dir = Path(persist_dir)
    if not base_dir.exists():
        return False
    marker_file = base_dir / "chroma.sqlite3"
    return marker_file.exists()


# 增量入库：把新文档块加进向量库，按固定 ID 去重。
#
# 参数：
#     doc_list: 要加进去的文档块。
#     embeddings: 向量化工具。
#     collection_name: 集合名。
#     persist_dir: 存向量库的目录。
#
# 返回：
#     (新增了几块, 跳过了几块重复的)。
#
# 注意：
#     - 集合已存在就打开接着加，不存在就新建。
#     - 只有真的加了新东西，才会让库版本号加一。
def add_chunks_to_index(doc_list, embeddings: Embeddings, collection_name: str, persist_dir: str):
    # 已存在的集合就打开，否则新建
    if vector_store_exists(persist_dir):
        store = open_vector_store(embeddings, collection_name, persist_dir)
        existing = store.get()
        existing_ids = set(existing["ids"])
    else:
        store = Chroma(
            embedding_function=embeddings,
            collection_name=collection_name,
            collection_metadata=_collection_metadata,
            persist_directory=persist_dir,
        )
        existing_ids = set()

    new_docs = []
    new_ids = []
    skipped_count = 0
    for one_doc in doc_list:
        doc_id = _make_doc_id(one_doc)
        if doc_id in existing_ids:
            skipped_count = skipped_count + 1
            continue
        new_docs.append(one_doc)
        new_ids.append(doc_id)
        existing_ids.add(doc_id)

    if new_docs:
        store.add_documents(documents=new_docs, ids=new_ids)
        bump_index_version()

    return len(new_docs), skipped_count
