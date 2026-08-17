"""测试：RAG 引擎（检索器阈值过滤、sources 返回、降级模式）。"""
from langchain_core.documents import Document

from src.rag.engine import _build_source_objs, _format_docs, _format_history
from src.rag.retriever import ThresholdRetriever


# ---- 纯函数测试 ----


def test_format_docs():
    docs = [
        Document(page_content="内容A", metadata={"source": "a.md"}),
        Document(page_content="内容B", metadata={"source": "b.md"}),
    ]
    result = _format_docs(docs)
    assert "[1] a.md" in result
    assert "[2] b.md" in result
    assert "内容A" in result
    assert "内容B" in result


def test_format_docs_empty():
    assert _format_docs([]) == "（无相关知识片段）"


def test_format_history_empty():
    assert _format_history([]) == "（无）"


def test_format_history():
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "您好"},
    ]
    result = _format_history(history)
    assert "user: 你好" in result
    assert "assistant: 您好" in result


def test_build_source_objs_dedup_by_source():
    """同一 source 的多个 chunk 合并为单个引用；编号唯一。"""
    docs = [
        Document(page_content="a", metadata={"source": "x.md", "score": 0.9}),
        Document(page_content="b", metadata={"source": "x.md", "score": 0.8}),
        Document(page_content="c", metadata={"source": "y.md", "score": 0.7}),
    ]
    objs = _build_source_objs(docs, with_content=True)
    assert len(objs) == 2  # 去重后只剩 2 个唯一 source
    assert [o["index"] for o in objs] == [1, 2]
    assert objs[0]["source"] == "x.md"
    assert objs[0]["chunk_count"] == 2  # x.md 合并了两个 chunk
    assert "a" in objs[0]["content"] and "b" in objs[0]["content"]
    assert objs[1]["source"] == "y.md"
    assert objs[1]["chunk_count"] == 1


def test_build_source_objs_without_content():
    """with_content=False（普通用户模式）不应泄露原文明文。"""
    docs = [Document(page_content="秘密内容", metadata={"source": "x.md", "score": 0.9})]
    objs = _build_source_objs(docs, with_content=False)
    assert len(objs) == 1
    assert "content" not in objs[0]
    assert objs[0]["source"] == "x.md"
    assert objs[0]["score"] == 0.9


# ---- 阈值检索器测试 ----


class _FakeStore:
    """模拟 Chroma store，返回固定相似度结果。"""

    def __init__(self, results: list[tuple[Document, float]]):
        self._results = results

    def similarity_search_with_relevance_scores(self, query: str, k: int):
        return self._results


def test_threshold_retriever_filters():
    store = _FakeStore(
        [
            (Document(page_content="高相似", metadata={"source": "a.md"}), 0.9),
            (Document(page_content="低相似", metadata={"source": "b.md"}), 0.3),
        ]
    )
    retriever = ThresholdRetriever(store=store, top_k=5, threshold=0.5)
    docs = retriever.invoke("测试")
    assert len(docs) == 1
    assert docs[0].page_content == "高相似"
    assert docs[0].metadata["score"] == 0.9


def test_threshold_retriever_all_below():
    store = _FakeStore(
        [(Document(page_content="无关", metadata={"source": "a.md"}), 0.1)]
    )
    retriever = ThresholdRetriever(store=store, top_k=5, threshold=0.5)
    docs = retriever.invoke("测试")
    assert docs == []
