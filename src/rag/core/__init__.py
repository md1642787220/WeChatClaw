"""RAG 核心子模块。

这个包把 RAG 流程拆成六个各管一摊、互不纠缠的模块：

- loader     读数据：把文件或字节流读成文字。
- splitter   切分：把长文按标题和长度切成小块。
- embedder   向量化：把文字块变成数字向量。
- indexer    建库：把文档块向量化后写进向量库。
- reranker   排序：对检索结果按相似度过滤、排序。
- store      存库：向量库的读写和存在性判断。

各模块用明确的函数接口互相调用，依赖方向从上往下（编排层 -> core 层），
不存在循环依赖。

Author: MADENG
Reviewer: Li Rongdong
"""
