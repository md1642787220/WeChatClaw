"""RAG 流水线（编排层）子包。

这个子包负责检索之后的编排逻辑，跟 core 层的具体实现解耦：

- context  上下文格式化：把检索结果整理成提示词上下文和来源列表。
- prompts  提示词模板：生成答案时用的 system / human 模板。

Author: MADENG
Reviewer: Li Rongdong
"""
