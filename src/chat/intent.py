"""意图识别：判断用户说的是闲聊，还是正经查资料。

- 闲聊类（你好/谢谢/再见等）→ 直接回复，不去查资料，更亲切。
- 业务类 → 走 RAG 检索。

实现方式：靠本地关键词 + 简单规则，不用调 LLM，毫秒级就能出结果。

Author: MADENG
Reviewer: Li Rongdong
"""
import re


# 闲聊关键词：问候、感谢、道别、自我介绍这几种
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


# 闲聊问题的预设回复
_GREETING_REPLIES = {
    "greeting": "您好！我是企业内部知识库助手，可以帮您查询公司政策、流程、福利等信息。请问您想了解什么？",
    "identity": "我是企业内部知识库助手，基于公司知识库回答员工常见问题。请问有什么可以帮您？",
    "thanks": "不客气！还有其他问题欢迎随时问我。",
    "goodbye": "好的，再见！如有问题随时回来找我。",
    "ok": "好的，还有其他问题吗？",
}


# 把闲聊正则列表拼成一个大正则（不区分大小写）。
def _build_greeting_regex():
    pattern = "|".join(_GREETING_PATTERNS)
    return re.compile(pattern, re.IGNORECASE)


_greeting_regex = _build_greeting_regex()


# 看看文字里有没有出现这些词（有一个就算命中）。
#
# 参数：
#     text: 要检查的文字。
#     keywords: 关键词列表。
#
# 返回：
#     命中返回 True，没命中返回 False。
def _has_any_keyword(text, keywords):
    for one_word in keywords:
        if one_word in text:
            return True
    return False


# 判断用户输入是闲聊还是查资料。
#
# 参数：
#     question: 用户输入。
#
# 返回：
#     (意图, 回复)：
#       - 意图：是 "chat"（闲聊）还是 "query"（查资料）。
#       - 回复：如果是闲聊，返回预设回复；查资料时是 None。
def classify_intent(question):
    raw_text = question.strip()
    # 空的，或者太长（可能是上下文里的复杂问题），都交给 RAG 处理
    if not raw_text:
        return ("query", None)
    if len(raw_text) > 200:
        return ("query", None)

    # 没匹配上闲聊模式，直接交给 RAG
    if not _greeting_regex.match(raw_text):
        return ("query", None)

    # 命中闲聊：细分是哪种，选对应回复
    # 顺序：身份 > 感谢 > 道别 > 确认 > 通用问候
    reply = _GREETING_REPLIES["greeting"]
    if _has_any_keyword(raw_text, ["你是", "你是谁", "介绍", "什么"]):
        reply = _GREETING_REPLIES["identity"]
    elif _has_any_keyword(raw_text, ["谢谢", "感谢", "多谢"]):
        reply = _GREETING_REPLIES["thanks"]
    elif _has_any_keyword(raw_text, ["再见", "拜拜", "晚安", "88"]):
        reply = _GREETING_REPLIES["goodbye"]
    elif re.search(r"好的|ok|明白", raw_text, re.IGNORECASE):
        reply = _GREETING_REPLIES["ok"]

    return ("chat", reply)
