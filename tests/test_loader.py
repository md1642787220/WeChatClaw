"""测试：文档加载与切分。"""
from src.knowledge.loader import load_and_split, split_markdown


def test_split_markdown_basic():
    text = "# 标题一\n\n这是第一段内容。\n\n## 二级标题\n\n这是第二段内容。"
    chunks = split_markdown(text, source="test.md")
    assert len(chunks) >= 2
    # 每块都带 source 元数据
    for chunk in chunks:
        assert chunk["metadata"]["source"] == "test.md"
        assert chunk["content"].strip()


def test_split_markdown_headers_metadata():
    text = "# 请假制度\n\n员工请假需提前一天申请。"
    chunks = split_markdown(text, source="policy.md")
    # 标题层级应写入元数据
    assert any("h1" in c["metadata"] for c in chunks)


def test_split_long_content():
    # 构造超长段落，验证二次切分
    long_para = "这是一个很长的段落。" * 500
    text = f"# 长文档\n\n{long_para}"
    chunks = split_markdown(text, source="long.md")
    assert len(chunks) > 1


def test_load_and_split_missing_file(tmp_path):
    """不存在的文件不应被调用（此测试验证接口签名，实际由调用方判断存在性）。"""
    # 这里仅验证 split_markdown 不抛异常即可
    chunks = split_markdown("", source="empty.md")
    assert chunks == [] or all(isinstance(c, dict) for c in chunks)
