"""RAG 子系统：检索增强生成的完整实现。

目录结构（依赖方向从上往下，单向、没有环）：

- engine.py     顶层编排：检索 -> 生成（含降级），对外提供「流式问答」。
- retriever.py  检索器：组合 core 层生成带阈值过滤的检索器（带缓存/预热）。
- llm.py        造聊天模型：用 OpenAI 兼容协议创建聊天模型。
- pipeline/     编排层：context（上下文格式化）、prompts（模板）。
- core/         核心子模块：loader / splitter / embedder / indexer / reranker / store。

Author: MADENG
Reviewer: Li Rongdong
"""
