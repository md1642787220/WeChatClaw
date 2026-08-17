"""索引构建模块：把「切分 -> 向量化 -> 存库」串起来完成建库。

这个模块是 RAG 流程的「入库编排层」，把切分、向量化、存库三个模块连起来，
给上层一个简单的建库/入库入口，不用关心底层细节。

依赖方向：indexer -> (splitter, embedder, store)，单向、没有环。

Author: MADENG
Reviewer: Li Rongdong
"""
from langchain_core.documents import Document

from src.rag.core.embedder import Embeddings
from src.rag.core.splitter import split_text_into_chunks
from src.rag.core.store import add_chunks_to_index, build_full_index


# 把一段文字切成 Document 列表（不向量化）。
#
# 参数：
#     text: 文档全文。
#     source: 来源文件名，写进元数据。
#
# 返回：
#     Document 列表，每个元素是一个文本块。
#
# 注意：
#     这个函数只切分，是「切分 -> 入库」的中间产物，
#     可以直接给「只看不存」的预览场景用。
def split_text_to_documents(text: str, source: str):
    chunk_dicts = split_text_into_chunks(text, source=source)
    doc_list = []
    for one_chunk in chunk_dicts:
        one_doc = Document(page_content=one_chunk["content"], metadata=one_chunk["metadata"])
        doc_list.append(one_doc)
    return doc_list


# 从一批文字全量建库。
#
# 参数：
#     text_list: 文字列表。
#     source_list: 跟 text_list 一样长的来源列表。
#     embeddings: 向量化工具。
#     collection_name: 集合名。
#     persist_dir: 存库目录。
#
# 返回：
#     Chroma 向量库实例。
#
# 注意：
#     全量建库会覆盖旧集合，想增量加请用「文档块入库」。
def build_index_from_texts(text_list, source_list, embeddings: Embeddings, collection_name: str, persist_dir: str):
    if len(text_list) != len(source_list):
        raise ValueError("text_list 和 source_list 长度必须一样")
    all_docs = []
    for i in range(len(text_list)):
        one_text = text_list[i]
        one_source = source_list[i]
        sub_docs = split_text_to_documents(one_text, one_source)
        for one_doc in sub_docs:
            all_docs.append(one_doc)
    return build_full_index(all_docs, embeddings, collection_name, persist_dir)


# 增量入库：把切好的 Document 列表写进向量库（按固定 ID 去重）。
#
# 参数：
#     doc_list: 切好的文档块。
#     embeddings: 向量化工具。
#     collection_name: 集合名。
#     persist_dir: 存库目录。
#
# 返回：
#     (新增块数, 跳过重复块数)。
def add_documents_to_index(doc_list, embeddings: Embeddings, collection_name: str, persist_dir: str):
    return add_chunks_to_index(doc_list, embeddings, collection_name, persist_dir)
