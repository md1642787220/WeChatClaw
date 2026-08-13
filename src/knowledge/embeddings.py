"""Embedding 封装：本地 sentence-transformers 模型，提供统一接口。"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    """基于 sentence-transformers 的本地 Embedding，兼容 LangChain Embeddings 接口。"""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()


def build_embeddings(model_name: str, device: str = "cpu") -> Embeddings:
    """构建 Embedding 实例。"""
    return LocalEmbeddings(model_name=model_name, device=device)
