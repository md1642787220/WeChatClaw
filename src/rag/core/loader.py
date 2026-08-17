"""数据加载模块：把文件或字节流读成一段文字。

这个模块只干一件事：把文件内容读出来，变成字符串。它不切分、不做向量，
跟别的模块各管各的，互不干扰。

支持两种输入：
- 文件路径：直接按 UTF-8 读。
- 原始字节：先按 UTF-8 解码，不行再试 GBK（中文老文件常用 GBK）。

Author: MADENG
Reviewer: Li Rongdong
"""
from pathlib import Path


# 读一个文本文件，返回里面的全部文字。
#
# 参数：
#     file_path: 文件路径（.md / .txt / .markdown 都行）。
#
# 返回：
#     文件里的全部文字。
#
# 注意：
#     只按 UTF-8 解码，如果编码不对会报错 UnicodeDecodeError，
#     要不要换别的编码，由调用的人自己决定。
def read_text_file(file_path: Path):
    return file_path.read_text(encoding="utf-8")


# 把原始字节变成文字。先试 UTF-8，不行再试 GBK。
#
# 参数：
#     raw_bytes: 文件的原始字节内容。
#
# 返回：
#     解码出来的文字。
#
# 注意：
#     GBK 这步用了 errors="replace"，遇到看不懂的字节会用占位符代替，
#     不会直接报错，保证后面的流程还能继续跑。
def decode_bytes_to_text(raw_bytes: bytes):
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("gbk", errors="replace")
