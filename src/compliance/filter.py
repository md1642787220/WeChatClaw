"""安全合规：敏感词过滤、输出脱敏。

Author: MADENG
Reviewer: Li Rongdong
"""
import re


# ---- 脱敏正则（都加了边界约束，避免误匹配） ----

# 手机号：1[3-9] 开头，11 位；前后不能是数字，避免在长数字串里误匹配
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")

# 身份证：18 位（末位可为 X），前 6 位地址码 + 8 位生日 + 3 位顺序 + 校验位；
# 前后不能是数字/字母，避免在更长串里误匹配
_ID_CARD_RE = re.compile(r"(?<![\dA-Za-z])\d{17}[\dXx](?![\dA-Za-z])")

# 邮箱：常见格式，local@domain，domain 至少含一个点
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# 银行卡号：13~19 位数字，前后不能是数字
_BANK_CARD_RE = re.compile(r"(?<!\d)\d{13,19}(?!\d)")

# 车牌号：含新能源（8 位），比如 粤A12345、京A·12345、粤AD12345（新能源）
_PLATE_RE = re.compile(
    r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]"
    r"[A-Z][A-Z0-9]{5,6}"
)

# 座机号：比如 010-12345678、0755-1234567
_LANDLINE_RE = re.compile(r"(?<!\d)(0\d{2,3}-?\d{7,8})(?!\d)")


# 工具函数：把字符串重复 N 次（用来生成 N 个 *）。
#
# 参数：
#     count: 重复次数。
#
# 返回：
#     重复后的字符串。
def _repeat_star(count):
    result = ""
    for _ in range(count):
        result = result + "*"
    return result


# 手机号脱敏：保留前 3 后 4，中间 ****。
def _mask_phone(match):
    raw = match.group()
    return raw[:3] + "****" + raw[-4:]


# 身份证脱敏：保留前 6 后 4，中间 ********。
def _mask_id_card(match):
    raw = match.group()
    return raw[:6] + "********" + raw[-4:]


# 邮箱脱敏：local 保留前 2，***，完整 domain。
def _mask_email(match):
    raw = match.group()
    # 按第一个 "@" 切分
    at_index = raw.find("@")
    if at_index < 0:
        return raw
    local_part = raw[:at_index]
    domain_part = raw[at_index + 1:]
    if len(local_part) <= 2:
        masked_local = local_part[:1] + "*"
    else:
        masked_local = local_part[:2] + "***"
    return masked_local + "@" + domain_part


# 银行卡脱敏：保留前 4 后 4，中间 ``**** ****``。
def _mask_bank_card(match):
    raw = match.group()
    return raw[:4] + " **** **** " + raw[-4:]


# 车牌脱敏：保留省份 + 末位，中间用 * 替换。
def _mask_plate(match):
    raw = match.group()
    middle_len = len(raw) - 3
    if middle_len < 0:
        middle_len = 0
    return raw[:2] + _repeat_star(middle_len) + raw[-1:]


# 座机号脱敏：区号 + -****- + 末 4 位。
def _mask_landline(match):
    raw = match.group()
    # 去掉 "-"
    digits = raw.replace("-", "")
    # 区号：以 "010" 开头就取 3 位（直辖市），否则取 4 位
    if digits.startswith("010"):
        area = digits[:3]
    else:
        area = digits[:4]
    tail = digits[-4:]
    return area + "-****-" + tail


# 检查有没有命中敏感词。
#
# 参数：
#     text: 要检查的文字。
#     sensitive_words: 敏感词列表。
#
# 返回：
#     命中的第一个敏感词；没命中返回 None。
def find_sensitive_word(text, sensitive_words):
    if not sensitive_words:
        return None
    for one_word in sensitive_words:
        if not one_word:
            continue
        if one_word in text:
            return one_word
    return None


# 给手机号、身份证、邮箱、银行卡、车牌、座机号打码。
#
# 参数：
#     text: 要脱敏的文字。
#
# 返回：
#     脱敏后的文字。
#
# 注意：
#     顺序：先处理带格式的（邮箱、车牌、座机），再处理纯数字
#     （身份证、银行卡、手机号），避免银行卡匹配覆盖身份证等场景。
def mask_sensitive_text(text):
    if not text:
        return text
    # 顺序：先处理带格式的（邮箱、车牌、座机），再处理纯数字（身份证、银行卡、手机号）
    text = _EMAIL_RE.sub(_mask_email, text)
    text = _PLATE_RE.sub(_mask_plate, text)
    text = _LANDLINE_RE.sub(_mask_landline, text)
    text = _ID_CARD_RE.sub(_mask_id_card, text)
    text = _BANK_CARD_RE.sub(_mask_bank_card, text)
    text = _PHONE_RE.sub(_mask_phone, text)
    return text


# 检查用户提问合不合规。
#
# 参数：
#     question: 用户提问。
#     sensitive_words: 敏感词列表。
#
# 返回：
#     (是否放行, 命中的敏感词)。放行 False 表示要拦截。
def check_compliance(question, sensitive_words):
    hit_word = find_sensitive_word(question, sensitive_words)
    if hit_word is not None:
        return False, hit_word
    return True, None
