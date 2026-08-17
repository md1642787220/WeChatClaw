"""RAG 核心子模块。

本包将 RAG 流程拆分为六个职责单一、低耦合的模块：

- ``loader``    数据加载模块：从文件/字节流读取文本内容。
- ``splitter``  文本分片模块：将长文本按标题层级与长度切分为块。
- ``embedder``  向量化模块：将文本块编码为向量。
- ``indexer``   索引构建模块：把文档块向量化并写入向量库。
- ``reranker``  排序模块：对检索结果按相似度过滤与排序。
- ``store``     持久化模块：向量库的加载、写入与存在性判断。

各模块通过明确的函数接口交互，依赖方向自上而下（编排层 -> core 层），
不存在循环依赖。

Author: MADENG
Reviewer: Li Rongdong
"""

from __future__ import annotations

__all__ = [
    "loader",
    "splitter",
    "embedder",
    "indexer",
    "reranker",
    "store",
]
