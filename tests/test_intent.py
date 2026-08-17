"""测试：意图识别。"""
from src.chat.intent import classify


def test_classify_greeting():
    intent, reply = classify("你好")
    assert intent == "chat"
    assert reply and "您好" in reply


def test_classify_greeting_with_punctuation():
    # 含中文标点
    intent, reply = classify("你好")  # 标点版本通过正则覆盖
    # 加标点再测一次，确保正则支持
    intent2, reply2 = classify("你好.")
    assert intent2 == "chat"
    assert reply2 is not None


def test_classify_hello_english():
    intent, _ = classify("Hello")
    assert intent == "chat"


def test_classify_identity():
    intent, reply = classify("你是谁？")
    assert intent == "chat"
    assert reply and "助手" in reply


def test_classify_thanks():
    intent, reply = classify("谢谢")
    assert intent == "chat"
    assert reply and "不客气" in reply


def test_classify_goodbye():
    intent, reply = classify("再见")
    assert intent == "chat"
    assert reply is not None


def test_classify_business_query():
    intent, reply = classify("年假有多少天？")
    assert intent == "query"
    assert reply is None


def test_classify_empty():
    intent, reply = classify("")
    assert intent == "query"


def test_classify_long_text():
    intent, reply = classify("x" * 250)
    assert intent == "query"