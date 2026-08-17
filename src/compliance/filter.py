"""安全合规：敏感词过滤、输出脱敏。"""
from __future__ import annotations

import re

# ---- 脱敏正则（均带边界约束，避免误匹配） ----

# 手机号：1[3-9] 开头，11 位；前后不能是数字，避免在长数字串中误匹配
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")

# 身份证：18 位（末位可为 X），前 6 位地址码 + 8 位生日 + 3 位顺序 + 校验位；
# 前后不能是数字/字母，避免在更长串中误匹配
_ID_CARD_RE = re.compile(r"(?<![\dA-Za-z])\d{17}[\dXx](?![\dA-Za-z])")

# 邮箱：常见格式，local@domain，domain 至少含一个点
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# 银行卡号：13~19 位数字，前后不能是数字
_BANK_CARD_RE = re.compile(r"(?<!\d)\d{13,19}(?!\d)")

# 车牌号：含新能源（8 位），如 粤A12345、京A·12345、粤AD12345（新能源）
_PLATE_RE = re.compile(
    r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]"
    r"[A-Z][A-Z0-9]{5,6}"
)

# 座机号：如 010-12345678、0755-1234567
_LANDLINE_RE = re.compile(r"(?<!\d)(0\d{2,3}-?\d{7,8})(?!\d)")


def _mask_phone(m: "re.Match[str]") -> str:
    s = m.group()
    return s[:3] + "****" + s[-4:]


def _mask_id_card(m: "re.Match[str]") -> str:
    s = m.group()
    return s[:6] + "********" + s[-4:]


def _mask_email(m: "re.Match[str]") -> str:
    s = m.group()
    local, _, domain = s.partition("@")
    if len(local) <= 2:
        masked_local = local[0] + "*"
    else:
        masked_local = local[:2] + "***"
    return f"{masked_local}@{domain}"


def _mask_bank_card(m: "re.Match[str]") -> str:
    s = m.group()
    return s[:4] + " **** **** " + s[-4:]


def _mask_plate(m: "re.Match[str]") -> str:
    s = m.group()
    return s[:2] + "*" * (len(s) - 3) + s[-1:]


def _mask_landline(m: "re.Match[str]") -> str:
    s = m.group()
    digits = s.replace("-", "")
    # 区号（前 3~4 位）+ 号码（后 7~8 位），中间打码
    area = digits[:3] if digits.startswith("010") else digits[:4]
    tail = digits[-4:]
    return f"{area}-****-{tail}"


def contains_sensitive(text: str, sensitive_words: list[str]) -> str | None:
    """检查是否命中敏感词，命中则返回命中的词，否则返回 None。"""
    if not sensitive_words:
        return None
    for word in sensitive_words:
        if word and word in text:
            return word
    return None


def desensitize(text: str) -> str:
    """对手机号、身份证、邮箱、银行卡、车牌、座机号做打码。

    顺序：先处理带格式的（邮箱、车牌、座机），再处理纯数字
    （身份证、银行卡、手机号），避免银行卡匹配覆盖身份证等场景。
    """
    if not text:
        return text
    text = _EMAIL_RE.sub(_mask_email, text)
    text = _PLATE_RE.sub(_mask_plate, text)
    text = _LANDLINE_RE.sub(_mask_landline, text)
    text = _ID_CARD_RE.sub(_mask_id_card, text)
    text = _BANK_CARD_RE.sub(_mask_bank_card, text)
    text = _PHONE_RE.sub(_mask_phone, text)
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
