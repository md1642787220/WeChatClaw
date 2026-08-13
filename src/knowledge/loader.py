"""文档加载与切分：支持 Markdown / txt，按标题层级递归切分。"""
from __future__ import annotations

from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Markdown 标题层级 → 元数据字段名
_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def load_text(path: Path) -> str:
    """读取纯文本 / Markdown 文件内容。"""
    return path.read_text(encoding="utf-8")


def split_markdown(text: str, source: str) -> list[dict]:
    """按 Markdown 标题层级切分，再按长度做二次切分。

    Args:
        text: 文档全文。
        source: 来源文件标识，写入元数据便于溯源。

    Returns:
        每个元素为 {"content": str, "metadata": dict}。
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    chunks = md_splitter.split_text(text)

    # 对过长的块做二次递归切分，保留层级元数据
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", " "],
    )

    result: list[dict] = []
    for chunk in chunks:
        if len(chunk.page_content) <= 1200:
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


def load_and_split(path: Path) -> list[dict]:
    """加载单个文件并切分，返回块列表。"""
    text = load_text(path)
    return split_markdown(text, source=str(path))
