"""上下文格式化模块：把搜到的文档整理成提示词上下文和来源列表。

这个模块把「检索结果 -> 文字/结构」这种纯格式化的活儿从引擎里拆出来，
生成答案和管理后台都能复用，不用重复写。

两条核心约定：
- 按来源文件名去重，同一个文件的多个块合并成一条引用。
- 引用编号从 1 开始连续排，跟提示词里的 [1][2] 一一对应。

Author: MADENG
Reviewer: Li Rongdong
"""


# 没搜到东西时写进上下文的提示文字，让模型知道「没命中」
NO_CONTEXT_HINT = "（无相关知识片段）"

# 来源信息缺了时用的兜底文字
UNKNOWN_SOURCE = "未知来源"


# 把若干块的正文用「---」分隔拼起来。
def _join_chunk_contents(chunk_list):
    parts = []
    for one_chunk in chunk_list:
        parts.append(one_chunk.page_content)
    return "\n---\n".join(parts)


# 从块里挑最小的 score（Chroma 里距离越小越像）。
def _get_best_score(chunk_list):
    best_score = None
    for one_chunk in chunk_list:
        # 先读 score，没有就退回读 distance
        score_value = one_chunk.metadata.get("score")
        if score_value is None:
            score_value = one_chunk.metadata.get("distance")
        if score_value is None:
            continue
        if best_score is None:
            best_score = score_value
            continue
        if score_value < best_score:
            best_score = score_value
    return best_score


# 按来源文件名去重，保留第一次出现的顺序。
#
# 参数：
#     doc_list: 搜到的文档列表（每个都带 metadata["source"]）。
#
# 返回：
#     [(编号, 来源, 块列表), ...]，编号从 1 开始连续排，
#     同一个文件的多个块合并到同一条引用下面。
def dedupe_by_source(doc_list):
    # seen: 来源 -> 它在结果列表里的位置
    seen_sources = {}
    result_list = []
    for one_doc in doc_list:
        source_value = one_doc.metadata.get("source")
        if source_value is None or source_value == "":
            source_value = UNKNOWN_SOURCE
        found_index = seen_sources.get(source_value)
        if found_index is not None:
            # 已经在结果里了：把 doc 追加到这个来源的块列表
            existing_entry = result_list[found_index]
            existing_entry[2].append(one_doc)
        else:
            # 第一次出现：加一条新记录
            new_index = len(result_list) + 1
            result_list.append([new_index, source_value, [one_doc]])
            seen_sources[source_value] = len(result_list) - 1
    return result_list


# 把搜到的文档块拼成带编号的上下文文字。
#
# 参数：
#     doc_list: 搜到的文档列表。
#
# 返回：
#     按来源去重后的编号上下文；空列表返回「没找到」提示。
def join_docs_as_context(doc_list):
    if not doc_list:
        return NO_CONTEXT_HINT
    parts = []
    for one_entry in dedupe_by_source(doc_list):
        index_number = one_entry[0]
        source_name = one_entry[1]
        chunk_list = one_entry[2]
        body_text = _join_chunk_contents(chunk_list)
        parts.append("[" + str(index_number) + "] " + source_name + "\n" + body_text)
    return "\n\n".join(parts)


# 整理成结构化的来源列表（按来源去重）。
#
# 参数：
#     doc_list: 搜到的文档列表。
#     with_content: True 就带上片段原文（管理员预览用）；
#         False 只给 编号/来源/分数/块数，不暴露原文（普通用户问答用）。
#
# 返回：
#     [{"index", "source", "score", "chunk_count", "content"?}, ...]
#     编号 index 跟「拼文档上下文」里的 [1][2] 对应。
def build_source_list(doc_list, with_content=True):
    result_list = []
    for one_entry in dedupe_by_source(doc_list):
        index_number = one_entry[0]
        source_name = one_entry[1]
        chunk_list = one_entry[2]
        best_score = _get_best_score(chunk_list)
        one_obj = {}
        one_obj["index"] = index_number
        one_obj["source"] = source_name
        one_obj["score"] = best_score
        one_obj["chunk_count"] = len(chunk_list)
        if with_content:
            one_obj["content"] = _join_chunk_contents(chunk_list)
        result_list.append(one_obj)
    return result_list


# 把历史对话拼成文字。
#
# 参数：
#     history: 历史消息列表，每项是 {"role": ..., "content": ...}。
#
# 返回：
#     每行一条「角色: 内容」的文字；没历史就返回「（无）」。
def join_history_as_text(history):
    if not history:
        return "（无）"
    lines = []
    for one_message in history:
        role = one_message["role"]
        content = one_message["content"]
        lines.append(role + ": " + content)
    return "\n".join(lines)
