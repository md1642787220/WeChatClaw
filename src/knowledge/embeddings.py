"""Embedding 封装：本地 sentence-transformers 模型，提供统一接口。

采用轻量中文模型 bge-small-zh-v1.5（约 95MB、512 维），
适合 CPU 离线部署，避免 BGE-M3 的 2GB+ 体积。
"""
from __future__ import annotations

import os
from pathlib import Path

from langchain_core.embeddings import Embeddings

# 国内环境默认走 HF 镜像，避免模型下载超时；已显式设置时尊重用户配置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 禁用 Xet 协议（其 CAS 服务器无法走镜像，会 401），强制走传统 HTTP 下载
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# 项目内本地模型目录（models/<org>__<name>）
_LOCAL_MODELS_DIR = Path("models")


def _resolve_local_path(model_name: str) -> str:
    """若 models/ 目录已存在对应模型，则返回本地路径，实现完全离线加载。"""
    local_dir = _LOCAL_MODELS_DIR / model_name.replace("/", "__")
    if local_dir.is_dir() and any(local_dir.iterdir()):
        return str(local_dir)
    return model_name


class LocalEmbeddings(Embeddings):
    """基于 sentence-transformers 的本地 Embedding，兼容 LangChain Embeddings 接口。"""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        # 优先加载项目内固化的本地模型，避免联网下载
        resolved = _resolve_local_path(model_name)
        self._model = SentenceTransformer(resolved, device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()


def build_embeddings(model_name: str, device: str = "cpu") -> Embeddings:
    """构建本地 Embedding 实例。"""
    return LocalEmbeddings(model_name=model_name, device=device)
