"""安全合规：敏感词过滤、输出脱敏。"""
from __future__ import annotations

import re

# 常见需脱敏字段
_PHONE_RE = re.compile(r"1[3-9]\d{9}")
_ID_CARD_RE = re.compile(r"\b\d{17}[\dXx]\b")


def contains_sensitive(text: str, sensitive_words: list[str]) -> str | None:
    """检查是否命中敏感词，命中则返回命中的词，否则返回 None。"""
    if not sensitive_words:
        return None
    for word in sensitive_words:
        if word and word in text:
            return word
    return None


def desensitize(text: str) -> str:
    """对手机号、身份证号做打码。"""
    text = _PHONE_RE.sub(lambda m: m.group()[:3] + "****" + m.group()[-4:], text)
    text = _ID_CARD_RE.sub(lambda m: m.group()[:6] + "********" + m.group()[-4:], text)
    return text


def check_compliance(
    question: str,
    sensitive_words: list[str],
) -> tuple[bool, str | None]:
    """对用户提问做合规检查。

    Returns:
        (是否放行, 命中敏感词)。放行为 False 时表示需拦截。
    """
    hit = contains_sensitive(question, sensitive_words)
    if hit:
        return False, hit
    return True, None
