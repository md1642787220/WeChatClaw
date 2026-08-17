"""文本分片模块：把长文章按标题层级和长度切成一小块一小块。

这个模块是 RAG 流程的第二步，进去的是文字，出来的是小块列表。
切分办法分两步：
1. 先按 Markdown 标题（# / ## / ###）切，顺便记下标题层级（h1/h2/h3）。
2. 对还是太长的块，再按长度切成更小的块，避免单块太长影响搜索精度。

Author: MADENG
Reviewer: Li Rongdong
"""
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


# Markdown 标题层级 -> 存进 metadata 的字段名
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# 单块长度上限：超过就再切一次
CHUNK_LENGTH_LIMIT = 1200

# 第二次切分的参数
FALLBACK_CHUNK_SIZE = 800
FALLBACK_CHUNK_OVERLAP = 100
FALLBACK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", " "]


# 把一个原始块转成结果的字典格式（带上来源）。
def _to_chunk_dict(raw_chunk, source):
    result = {}
    result["content"] = raw_chunk.page_content
    merged_metadata = {}
    for one_key, one_value in raw_chunk.metadata.items():
        merged_metadata[one_key] = one_value
    merged_metadata["source"] = source
    result["metadata"] = merged_metadata
    return result


# 按 Markdown 标题切，再对过长的块做第二次切分。
#
# 参数：
#     text: 文档全文。
#     source: 来源文件名，会写进每个块的 metadata["source"]，方便追来源。
#
# 返回：
#     块列表，每个元素是 {"content": 文字, "metadata": 字典}。
#
# 注意：
#     - 每个块都会带上 source，方便后面去重和展示来源。
#     - 标题切出来的层级信息（h1/h2/h3）会跟着第二次切分一起保留。
def split_text_into_chunks(text: str, source: str):
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    raw_chunks = header_splitter.split_text(text)

    # 对太长的块做第二次切分，保留标题层级
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=FALLBACK_CHUNK_SIZE,
        chunk_overlap=FALLBACK_CHUNK_OVERLAP,
        separators=FALLBACK_SEPARATORS,
    )

    result_chunks = []
    for one_chunk in raw_chunks:
        chunk_length = len(one_chunk.page_content)
        if chunk_length <= CHUNK_LENGTH_LIMIT:
            result_chunks.append(_to_chunk_dict(one_chunk, source))
        else:
            sub_texts = fallback_splitter.split_text(one_chunk.page_content)
            for sub_text in sub_texts:
                # 第二次切分后 metadata 丢了，用父块的 metadata 补上
                rebuilt_chunk = Document(page_content=sub_text, metadata=one_chunk.metadata)
                result_chunks.append(_to_chunk_dict(rebuilt_chunk, source))
    return result_chunks
