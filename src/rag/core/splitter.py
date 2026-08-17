"""文本分片模块：将长文本按 Markdown 标题层级与长度切分为块。

本模块是 RAG 流程的第二步，输入为文本字符串，输出为块列表。
切分策略：
1. 先用 Markdown 标题层级（# / ## / ###）切分，保留层级元数据（h1/h2/h3）。
2. 对超过阈值的块，再按长度递归二次切分，避免单块过长影响检索精度。

Author: MADENG
Reviewer: Li Rongdong
"""

from __future__ import annotations

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Markdown 标题层级 -> 元数据字段名（写入每个块的 metadata）
_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# 单块长度阈值：超过该值则触发二次切分
_CHUNK_LENGTH_LIMIT = 1200

# 二次切分参数
_FALLBACK_CHUNK_SIZE = 800
_FALLBACK_CHUNK_OVERLAP = 100
_FALLBACK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", " "]


# 按 Markdown 标题层级切分，再对过长块做二次切分。
#
# Args:
#     text: 文档全文。
#     source: 来源文件标识，会写入每个块的 ``metadata["source"]`` 便于溯源。
#
# Returns:
#     块列表，每个元素为 ``{"content": str, "metadata": dict}``。
#
# Notes:
#     - 每个块都会携带 ``source`` 元数据，用于后续去重与来源展示。
#     - 标题切分产生的层级元数据（h1/h2/h3）会随二次切分一并保留。
def split_markdown(text: str, source: str) -> list[dict]:
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    chunks = md_splitter.split_text(text)

    # 对过长的块做二次递归切分，保留层级元数据
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_FALLBACK_CHUNK_SIZE,
        chunk_overlap=_FALLBACK_CHUNK_OVERLAP,
        separators=_FALLBACK_SEPARATORS,
    )

    result: list[dict] = []
    for chunk in chunks:
        if len(chunk.page_content) <= _CHUNK_LENGTH_LIMIT:
            result.append(
                {
                    "content": chunk.page_content,
                    "metadata": {**chunk.metadata, "source": source},
                }
            )
        else:
            for sub in fallback_splitter.split_text(chunk.page_content):
                result.append(
                    {
                        "content": sub,
                        "metadata": {**chunk.metadata, "source": source},
                    }
                )
    return result
