"""意图识别：判断用户输入是闲聊还是业务查询。

- 闲聊类（你好/谢谢/再见等）→ 直接回复，不检索，响应更友好。
- 业务类 → 走 RAG 检索。

实现：基于本地关键词 + 简单规则，无需 LLM 调用，毫秒级响应。

Author: MADENG
Reviewer: Li Rongdong
"""
from __future__ import annotations

import re

# 闲聊关键词：问候、感谢、道别、自我介绍类提问
_GREETING_PATTERNS = [
    r"^你好[!?？!。，,.\s]*$",
    r"^您好[!?？!。，,.\s]*$",
    r"^hi[!?？!。，,.\s]*$",
    r"^hello[!?？!。，,.\s]*$",
    r"^嗨[!?？!。，,.\s]*$",
    r"^在吗[!?？!。，,.\s]*$",
    r"^在不在[!?？!。，,.\s]*$",
    r"^你是谁[!?？!。，,.\s]*$",
    r"^你是什么[!?？!。，,.\s]*$",
    r"^你能做什么[!?？!。，,.\s]*$",
    r"^你是做什么的[!?？!。，,.\s]*$",
    r"^你是哪位[!?？!。，,.\s]*$",
    r"^介绍下你自己[!?？!。，,.\s]*$",
    r"^自我介绍[!?？!。，,.\s]*$",
    r"^谢谢[!?？!。，,.\s]*$",
    r"^感谢[!?？!。，,.\s]*$",
    r"^多谢[!?？!。，,.\s]*$",
    r"^好的[!?？!。，,.\s]*$",
    r"^ok[!?？!。，,.\s]*$",
    r"^明白了[!?？!。，,.\s]*$",
    r"^好的我知道了[!?？!。，,.\s]*$",
    r"^再见[!?？!。，,.\s]*$",
    r"^拜拜[!?？!。，,.\s]*$",
    r"^88[!?？!。，,.\s]*$",
    r"^晚安[!?？!。，,.\s]*$",
]

_GREETING_RE = re.compile("|".join(_GREETING_PATTERNS), re.IGNORECASE)


# 闲聊类问题的预设回复
_GREETING_REPLIES: dict[str, str] = {
    "greeting": "您好！我是企业内部知识库助手，可以帮您查询公司政策、流程、福利等信息。请问您想了解什么？",
    "identity": "我是企业内部知识库助手，基于公司知识库回答员工常见问题。请问有什么可以帮您？",
    "thanks": "不客气！还有其他问题欢迎随时问我。",
    "goodbye": "好的，再见！如有问题随时回来找我。",
    "ok": "好的，还有其他问题吗？",
}


# 分类用户输入。
#
# Args:
#     question: 用户输入。
#
# Returns:
#     ``(intent, reply)`` 二元组：
#       - intent: ``"chat"``（闲聊）/ ``"query"``（业务查询）
#       - reply: 若为闲聊，返回预设回复；业务查询时为 ``None``
def classify(question: str) -> tuple[str, str | None]:
    q = question.strip()
    if not q or len(q) > 200:
        # 空或过长（可能是上下文中的复杂问题），交给 RAG 处理
        return ("query", None)

    if not _GREETING_RE.match(q):
        return ("query", None)

    # 细化闲聊子类
    if re.search(r"你是|你是谁|介绍|什么", q):
        reply = _GREETING_REPLIES["identity"]
    elif re.search(r"谢谢|感谢|多谢", q):
        reply = _GREETING_REPLIES["thanks"]
    elif re.search(r"再见|拜拜|晚安|88", q):
        reply = _GREETING_REPLIES["goodbye"]
    elif re.search(r"好的|ok|明白", q, re.IGNORECASE):
        reply = _GREETING_REPLIES["ok"]
    else:
        reply = _GREETING_REPLIES["greeting"]

    return ("chat", reply)