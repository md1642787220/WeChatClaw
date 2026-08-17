"""数据加载模块：从文件路径或原始字节流读取文档文本。

本模块是 RAG 流程的第一步，仅负责「把内容读成字符串」，不做切分、
不做向量化，与其它模块保持解耦。

支持两类输入：
- 文件路径（``Path``）：直接以 UTF-8 读取。
- 原始字节（``bytes``）：按 UTF-8 解码，失败时回退 GBK（兼容常见中文编码）。

Author: MADENG
Reviewer: Li Rongdong
"""

from __future__ import annotations

from pathlib import Path


# 从文件路径读取文本内容（UTF-8）。
#
# Args:
#     path: 文档文件路径（.md / .txt / .markdown）。
#
# Returns:
#     文件全文（字符串）。
#
# Notes:
#     仅做 UTF-8 解码，遇非法编码会抛 ``UnicodeDecodeError``，
#     由调用方决定是否回退其它编码。
def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# 将原始字节流解码为文本（UTF-8 优先，失败回退 GBK）。
#
# Args:
#     raw: 文档的原始字节内容。
#
# Returns:
#     解码后的文本字符串。
#
# Notes:
#     GBK 回退使用 ``errors="replace"``，非法字节会替换为占位符，
#     不会抛异常，保证上游流程可继续执行。
def decode_bytes(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gbk", errors="replace")
