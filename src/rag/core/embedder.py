"""向量化模块：把一段段文字变成一串数字（向量）。

简单说，就是用一个本地模型，把文字转成数字串，方便后面做相似度比较。
用的模型是 bge-small-zh-v1.5，只有 95MB、512 维，很轻量，CPU 上就能跑。

几个关键点：
- 优先用项目里 models/ 目录已经下载好的模型，这样完全不用联网。
- 向量做了归一化（就是把长度压到一样），配合 Chroma 的余弦相似度来算「像不像」。
- 实现了 LangChain 的 Embeddings 接口，向量库和检索器都能直接用。

Author: MADENG
Reviewer: Li Rongdong
"""
import os
from pathlib import Path

from langchain_core.embeddings import Embeddings

# 国内环境默认走 HF 镜像，避免下载超时；用户自己设过的话就听用户的
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 关掉 Xet 协议（它的服务器不能走镜像，会报 401），改走普通 HTTP 下载
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# 项目里放本地模型的目录（models/<组织名>__<模型名>）
local_models_folder = Path("models")


# 解析模型路径：优先返回项目里已经下载好的本地目录。
#
# 参数：
#     model_name: 模型名（比如 BAAI/bge-small-zh-v1.5）。
#
# 返回：
#     如果 models/ 下有对应的模型目录，就返回目录路径；否则原样返回模型名。
#
# 注意：
#     只有目录非空才算有效，避免命中残留的空目录导致加载失败。
def _find_local_model_path(model_name: str):
    local_dir = local_models_folder / model_name.replace("/", "__")
    if local_dir.is_dir():
        # 看看里面有没有东西
        has_anything = False
        for _ in local_dir.iterdir():
            has_anything = True
            break
        if has_anything:
            return str(local_dir)
    return model_name


# 本地向量化工具，兼容 LangChain 的 Embeddings 接口。
class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        # 优先加载本地模型，避免联网
        resolved_path = _find_local_model_path(model_name)
        self._model = SentenceTransformer(resolved_path, device=device)

    # 把一批文字都变成向量。
    #
    # 参数：
    #     text_list: 要处理的文字列表。
    #
    # 返回：
    #     和输入一样长的向量列表，每个向量是一个 float 列表（已经归一化）。
    def embed_documents(self, text_list):
        raw_vectors = self._model.encode(text_list, normalize_embeddings=True)
        result_list = []
        for one_vector in raw_vectors:
            result_list.append(one_vector.tolist())
        return result_list

    # 把一条查询文字变成向量。
    def embed_query(self, text: str):
        one_vector = self._model.encode(text, normalize_embeddings=True)
        return one_vector.tolist()


# 造一个本地向量化工具（工厂函数）。
#
# 参数：
#     model_name: 模型名。
#     device: 用什么设备算，cpu 或 cuda。
#
# 返回：
#     一个本地向量化实例。
def make_embedder(model_name: str, device: str = "cpu"):
    return LocalEmbeddings(model_name=model_name, device=device)
