"""排序模块：对检索出来的结果按相似度过滤、排序。

这个模块管「搜完之后、生成答案之前」的一环，具体干三件事：
- 按相似度阈值，把不够像的结果丢掉；
- 把命中的结果按相似度从高到低排好；
- 把相似度分数写进每个文档的 metadata，方便前面展示。

注意：这个模块只做「过滤/排序」这类纯逻辑，不碰向量库和模型，
跟别的模块解耦，以后想换成更复杂的重排策略（比如 cross-encoder）也好替换。

Author: MADENG
Reviewer: Li Rongdong
"""
from langchain_core.documents import Document


# 相似度分数写进 metadata 时用的键名，前面统一读这个
SCORE_KEY = "score"


# 按相似度阈值过滤结果，并把分数写进 metadata。
#
# 参数：
#     doc_list: 搜到的文档列表（和 score_list 一样长）。
#     score_list: 每个文档对应的相似度分数。
#     threshold: 相似度阈值，比它低的文档丢掉。
#
# 返回：
#     (文档, 分数) 列表，只留 >= 阈值的，分数已四舍五入到 4 位小数，
#     并写进了 doc.metadata["score"]。
#
# 注意：
#     不会改原来的 doc_list / score_list，而是返回一个新列表；
#     但会直接往 doc 的 metadata 里写分数（方便后面直接读）。
def filter_by_score_threshold(doc_list, score_list, threshold: float):
    if len(doc_list) != len(score_list):
        raise ValueError("doc_list 和 score_list 长度必须一样")
    result_list = []
    for i in range(len(doc_list)):
        one_doc = doc_list[i]
        one_score = score_list[i]
        if one_score < threshold:
            continue
        one_doc.metadata[SCORE_KEY] = round(one_score, 4)
        result_list.append((one_doc, one_score))
    return result_list


# 给 (文档, 分数) 列表按分数排序（稳定排序）。
def _sort_pairs(pair_list, reverse):
    return sorted(pair_list, key=lambda p: p[1], reverse=reverse)


# 按相似度分数排序，返回文档列表（不含分数）。
#
# 参数：
#     pair_list: (文档, 分数) 列表。
#     reverse: True 表示从高到低（默认），False 表示从低到高。
#
# 返回：
#     排好序的文档列表（不含分数）。
def sort_by_score(pair_list, reverse=True):
    sorted_pairs = _sort_pairs(pair_list, reverse)
    result_docs = []
    for one_pair in sorted_pairs:
        result_docs.append(one_pair[0])
    return result_docs


# 「过滤 + 排序」一起做，是检索器常用的便捷入口。
#
# 参数：
#     doc_list: 搜到的文档列表。
#     score_list: 对应分数。
#     threshold: 相似度阈值。
#
# 返回：
#     过滤后按相似度从高到低排好的文档列表。
def filter_and_sort(doc_list, score_list, threshold: float):
    filtered = filter_by_score_threshold(doc_list, score_list, threshold)
    return sort_by_score(filtered)
