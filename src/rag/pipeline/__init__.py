"""RAG 流水线（编排层）子包。

本子包承载检索后的编排逻辑，与 core 层的具体实现解耦：

- ``context``  上下文格式化：将检索结果组织为 Prompt 上下文与来源列表。
- ``prompts``  Prompt 模板：生成阶段使用的 system / human 模板。

Author: MADENG
Reviewer: Li Rongdong
"""

from __future__ import annotations

__all__ = ["context", "prompts"]
