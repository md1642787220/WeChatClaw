"""检索器模块：从向量库里造一个带相似度阈值过滤的检索器。

这个模块是 RAG 流程的「检索入口」，把 core 层的向量化、存库、排序模块组合起来，
给上层提供一个 BaseRetriever 接口。

还带一个模块级缓存：启动时用「预热检索器」先把向量化模型和向量库加载好，
后面的请求直接复用，不用每次请求都重新加载模型。

Author: MADENG
Reviewer: Li Rongdong
"""
import logging

from langchain_core.retrievers import BaseRetriever

from src.config import Settings
from src.rag.core.embedder import make_embedder
from src.rag.core.reranker import filter_by_score_threshold
from src.rag.core.store import open_vector_store, vector_store_exists

logger = logging.getLogger(__name__)

# 模块级缓存：检索器单例，避免反复加载向量化模型
# 键由 存库目录/集合名/模型/设备/取几条/阈值 拼成
_retriever_cache = {}


# 空检索器：向量库还没建好的时候用，返回空结果。
#
# 注意：
#     这个类不加载向量化模型，避免服务启动时因为库不存在而触发下载。
class _EmptyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str):
        return []


# 按相似度阈值过滤的检索器。
#
# 属性：
#     store: Chroma 向量库实例。
#     top_k: 最多返回几条。
#     threshold: 相似度阈值（余弦相似度，范围 0 到 1）。
#
# 注意：
#     内部先调向量库做相似度搜索，再用排序模块按阈值过滤。
#     过滤后的分数会写进 doc.metadata["score"]，给前面展示来源用。
class ThresholdRetriever(BaseRetriever):
    store: object
    top_k: int
    threshold: float

    def _get_relevant_documents(self, query: str):
        # 这个函数返回 (文档, 分数) 列表
        search_results = self.store.similarity_search_with_relevance_scores(query, k=self.top_k)
        doc_list = []
        score_list = []
        for one_pair in search_results:
            doc_list.append(one_pair[0])
            score_list.append(one_pair[1])
        # 交给排序模块按阈值过滤（分数会同步写进 metadata["score"]）
        filtered_pairs = filter_by_score_threshold(doc_list, score_list, self.threshold)
        final_docs = []
        for one_pair in filtered_pairs:
            final_docs.append(one_pair[0])
        return final_docs


# 根据配置的关键字段拼一个缓存键。
#
# 参数：
#     settings: 全局配置。
#
# 返回：
#     由 存库目录、集合名、模型名/设备、检索参数 拼出来的字符串。
#
# 注意：
#     任何一个关键字段变了，键就不一样，从而触发检索器重建。
def _make_cache_key(settings: Settings):
    parts = [
        settings.vector_db.persist_dir,
        settings.vector_db.collection,
        settings.embedding.model,
        settings.embedding.device,
        str(settings.retrieval.top_k),
        str(settings.retrieval.threshold),
    ]
    return "|".join(parts)


# 造一个检索器（带缓存）。
#
# 参数：
#     settings: 全局配置。
#     use_cache: True 就复用缓存的检索器，避免反复加载模型。
#
# 返回：
#     - 向量库建好了：返回带阈值过滤的检索器。
#     - 还没建：返回空占位检索器（不加载向量化模型）。
#
# 注意：
#     第一次构建要加载向量化模型，比较慢（秒级），
#     建议启动时用「预热检索器」先加载好。
def make_retriever(settings: Settings, use_cache=True):
    if not vector_store_exists(settings.vector_db.persist_dir, settings.vector_db.collection):
        logger.info("向量库还没建好，先用空检索器（请先入库）")
        return _EmptyRetriever()

    cache_key = _make_cache_key(settings)
    if use_cache:
        cached_retriever = _retriever_cache.get(cache_key)
        if cached_retriever is not None:
            return cached_retriever

    embedder = make_embedder(
        model_name=settings.embedding.model,
        device=settings.embedding.device,
    )
    store = open_vector_store(
        embeddings=embedder,
        collection_name=settings.vector_db.collection,
        persist_dir=settings.vector_db.persist_dir,
    )
    retriever = ThresholdRetriever(
        store=store,
        top_k=settings.retrieval.top_k,
        threshold=settings.retrieval.threshold,
    )
    if use_cache:
        _retriever_cache[cache_key] = retriever
    return retriever


# 启动时预热：先把向量化模型和向量库加载好，填进缓存。
#
# 参数：
#     settings: 全局配置。
#
# 注意：
#     在 FastAPI 启动时调用，让第一个 /chat、/kb/search 请求不用再扛模型加载的开销。
#     向量库还没建的话直接跳过，降级成第一次检索时再加载。
def warm_up_retriever(settings: Settings):
    if not vector_store_exists(settings.vector_db.persist_dir, settings.vector_db.collection):
        logger.info("向量库还没建好，跳过预热（入库后重启或首次检索时会自动加载）")
        return
    logger.info(
        "预热知识库：加载向量化模型和向量库（persist_dir=%s, collection=%s）...",
        settings.vector_db.persist_dir,
        settings.vector_db.collection,
    )
    make_retriever(settings, use_cache=True)
    logger.info("知识库预热完成，后面的请求会直接复用缓存的检索器。")


# 向量库内容变了（入库）后调这个，清掉缓存好下次重建。
#
# 注意：
#     调了之后，下一个检索请求会重新加载向量库和向量化模型，
#     这样就能反映最新入库的数据。
def clear_retriever_cache():
    _retriever_cache.clear()
