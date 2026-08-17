"""向量化模块：将文本块编码为向量（本地 sentence-transformers 模型）。

本模块是 RAG 流程的第三步，输入为文本列表，输出为向量列表。
采用轻量中文模型 bge-small-zh-v1.5（约 95MB、512 维），
适合 CPU 离线部署，避免 BGE-M3 的 2GB+ 体积。

设计要点：
- 优先加载项目内 ``models/`` 目录固化的本地模型，实现完全离线。
- 向量做 L2 归一化，配合 Chroma 余弦相似度检索（值域 [0,1]，越大越相似）。
- 兼容 LangChain ``Embeddings`` 接口，供向量库与检索器直接调用。

Author: MADENG
Reviewer: Li Rongdong
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


# 解析模型路径：优先返回项目内固化的本地目录。
#
# Args:
#     model_name: 模型名（如 ``BAAI/bge-small-zh-v1.5``）。
#
# Returns:
#     若 ``models/`` 下存在对应模型目录则返回其路径，否则原样返回模型名。
#
# Notes:
#     仅当目录非空时视为有效，避免命中残留空目录导致加载失败。
def _resolve_local_path(model_name: str) -> str:
    local_dir = _LOCAL_MODELS_DIR / model_name.replace("/", "__")
    if local_dir.is_dir() and any(local_dir.iterdir()):
        return str(local_dir)
    return model_name


# 基于 sentence-transformers 的本地 Embedding，兼容 LangChain 接口。
#
# Attributes:
#     _model: 已加载的 SentenceTransformer 实例。
#
# Notes:
#     模型在 ``__init__`` 时加载，成本较高（数百 MB、秒级）。
#     上层应通过缓存复用实例，避免每次请求重复加载。
class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        # 优先加载项目内固化的本地模型，避免联网下载
        resolved = _resolve_local_path(model_name)
        self._model = SentenceTransformer(resolved, device=device)

    # 批量向量化文档块。
    #
    # Args:
    #     texts: 待编码的文本列表。
    #
    # Returns:
    #     与输入等长的向量列表，每个向量为 ``list[float]``（已归一化）。
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    # 向量化单条查询。
    #
    # Args:
    #     text: 查询文本。
    #
    # Returns:
    #     归一化后的查询向量。
    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()


# 构建本地 Embedding 实例（工厂函数）。
#
# Args:
#     model_name: 模型名（如 ``BAAI/bge-small-zh-v1.5``）。
#     device: 计算设备（``cpu`` / ``cuda``）。
#
# Returns:
#     实现 LangChain ``Embeddings`` 接口的本地向量化实例。
def build_embeddings(model_name: str, device: str = "cpu") -> Embeddings:
    return LocalEmbeddings(model_name=model_name, device=device)
