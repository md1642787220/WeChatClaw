"""RAG 子系统：检索增强生成的完整实现。

目录结构（依赖方向自上而下，单向无环）：

- ``engine.py``   顶层编排：检索 -> 生成（含降级），对外暴露 stream_chat。
- ``retriever.py`` 检索器：组合 core 层生成带阈值过滤的检索器（含缓存/预热）。
- ``llm.py``      LLM 构建：基于 OpenAI 兼容协议创建 ChatModel。
- ``pipeline/``   编排层：context（上下文格式化）、prompts（模板）。
- ``core/``       核心子模块：loader / splitter / embedder / indexer / reranker / store。

Author: MADENG
Reviewer: Li Rongdong
"""
