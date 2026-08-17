"""测试：安全合规（敏感词过滤、脱敏）。"""
from src.compliance.filter import check_compliance, contains_sensitive, desensitize


def test_contains_sensitive_hit():
    assert contains_sensitive("请问底价是多少", ["底价", "内部"]) == "底价"


def test_contains_sensitive_no_hit():
    assert contains_sensitive("产品如何使用", ["底价", "内部"]) is None


def test_contains_sensitive_empty_words():
    assert contains_sensitive("随便问问", []) is None


def test_check_compliance_block():
    ok, hit = check_compliance("我想知道内部员工名单", ["内部", "员工"])
    assert ok is False
    assert hit == "内部"


def test_check_compliance_pass():
    ok, hit = check_compliance("请假流程是怎样的", ["内部", "员工"])
    assert ok is True
    assert hit is None


def test_desensitize_phone():
    text = "联系电话 13812345678，欢迎咨询"
    result = desensitize(text)
    assert "13812345678" not in result
    assert "138****5678" in result


def test_desensitize_phone_not_in_long_number():
    # 12 位数字（比手机号多一位）不应被手机号规则误匹配
    text = "编号 13812345678"
    result = desensitize(text)
    assert "138****5678" in result  # 11 位正常脱敏

    text2 = "编号 138123456789"
    result2 = desensitize(text2)
    assert "138123456789" in result2  # 12 位保持原样（非手机号）


def test_desensitize_id_card():
    text = "身份证 110101199001011234 已登记"
    result = desensitize(text)
    assert "110101199001011234" not in result
    assert "110101********1234" in result


def test_desensitize_email():
    text = "联系邮箱 test@example.com 或 admin@foo.cn"
    result = desensitize(text)
    assert "test@example.com" not in result
    assert "admin@foo.cn" not in result


def test_desensitize_bank_card():
    text = "银行卡号 6222021234567890 请核对"
    result = desensitize(text)
    assert "6222021234567890" not in result


def test_desensitize_plate():
    text = "车牌 粤A12345 已登记"
    result = desensitize(text)
    assert "粤A12345" not in result


def test_desensitize_landline():
    text = "座机 010-12345678"
    result = desensitize(text)
    assert "010-12345678" not in result


def test_desensitize_empty():
    assert desensitize("") == ""
