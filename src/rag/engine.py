"""RAG 引擎：把「检索 -> 生成答案」串起来（没配 LLM 时自动降级）。

统一返回结构：{"answer": 文字, "sources": 来源列表}。
不管是走 LLM 生成，还是降级模式，都会把命中的知识来源返回出来。

流式能力：「流式问答」返回一个生成器，一块一块地产出答案，
结束的时候再给出 token 用量和来源。

这个模块是 RAG 流程的「编排层」，依赖关系：
    engine -> pipeline(上下文/提示词) + retriever -> core(...)
它自己不干检索/切分/向量化这些具体活儿，只负责把各个模块串起来调用。

Author: MADENG
Reviewer: Li Rongdong
"""
import logging
import time

from langchain_core.output_parsers import StrOutputParser

from src.chat.intent import classify_intent
from src.config import Settings
from src.rag.llm import make_chat_model
from src.rag.pipeline.context import build_source_list, join_docs_as_context, join_history_as_text
from src.rag.pipeline.prompts import build_prompt
from src.rag.retriever import make_retriever

logger = logging.getLogger(__name__)


# 把没命中的文档列表拼成一段带编号的降级回复。
#
# 参数：
#     doc_list: 检索结果（可能是空的）。
#
# 返回：
#     拼好的整段文字；空列表返回「没找到」提示。
def _build_fallback_answer(doc_list):
    if not doc_list:
        return "当前知识库中未找到相关信息。（未配置 LLM，无法生成友好回复）"
    parts = []
    index_number = 1
    for one_doc in doc_list:
        source = one_doc.metadata.get("source", "未知")
        content = one_doc.page_content
        parts.append("[" + str(index_number) + "] " + source + "\n" + content)
        index_number = index_number + 1
    joined_text = "\n\n".join(parts)
    return "（未配置 LLM，以下为命中的知识片段原文）\n\n" + joined_text


# 造一个全是 0 的 token 统计字典。
def _zero_token_usage():
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# 从 LLM 的一块输出里抠 token 数。
# 兼容 input_tokens / output_tokens 和旧的 prompt_tokens / completion_tokens 两个字段名。
#
# 参数：
#     chunk: LLM 流式输出的一块（可能带 usage_metadata）。
#     current_prompt: 当前 prompt_tokens 累计值。
#     current_completion: 当前 completion_tokens 累计值。
#
# 返回：
#     (prompt_tokens, completion_tokens)。
def _extract_token_usage(chunk, current_prompt, current_completion):
    usage = getattr(chunk, "usage_metadata", None)
    if not usage:
        return current_prompt, current_completion
    prompt_token = usage.get("input_tokens")
    if prompt_token is None:
        prompt_token = usage.get("prompt_tokens")
    if prompt_token is not None:
        current_prompt = prompt_token
    completion_token = usage.get("output_tokens")
    if completion_token is None:
        completion_token = usage.get("completion_tokens")
    if completion_token is not None:
        current_completion = completion_token
    return current_prompt, current_completion


# 走 LLM 生成：检索 + 提示词 + LLM + 解析文字，并带上来源。
#
# 参数：
#     retriever: 检索器实例。
#     llm: LLM 实例。
#     prompt_template: 提示词模板。
#     question: 用户问题。
#     history: 历史对话列表。
#
# 返回：
#     {"answer": 文字, "sources": 列表}
def _run_llm_with_sources(retriever, llm, prompt_template, question, history):
    doc_list = retriever.invoke(question)
    context_text = join_docs_as_context(doc_list)  # 没命中的时候是 "(无相关知识片段)"
    history_text = join_history_as_text(history)
    messages = prompt_template.format_messages(
        context=context_text,
        history=history_text,
        question=question,
    )
    # LLM 调用 -> 解析输出
    raw_output = llm.invoke(messages)
    parser = StrOutputParser()
    answer_text = parser.invoke(raw_output)
    source_list = build_source_list(doc_list)
    return {"answer": answer_text, "sources": source_list}


# 降级模式：返回命中的片段原文和出处，不调 LLM。
#
# 参数：
#     retriever: 检索器实例。
#     question: 用户问题。
#
# 返回：
#     {"answer": 文字, "sources": 列表}
def _run_fallback(retriever, question):
    doc_list = retriever.invoke(question)
    answer_text = _build_fallback_answer(doc_list)
    source_list = build_source_list(doc_list)
    return {"answer": answer_text, "sources": source_list}


# 搭一条问答流水线。
#
# 参数：
#     settings: 全局配置。
#
# 返回：
#     - 配了 LLM Key：返回「检索 -> 提示词 -> LLM -> 文字」流水线。
#     - 没配 LLM Key：返回「检索 -> 拼片段」降级流水线。
#
# 注意：
#     返回的流水线可以当 chain(payload) 来调，
#     其中 payload = {"question": 文字, "history": 列表}。
def build_rag_pipeline(settings: Settings):
    retriever = make_retriever(settings)

    if settings.llm.available:
        chat_model = make_chat_model(settings.llm)
        prompt_template = build_prompt(settings.llm.system_role)

        def run_with_sources(payload):
            question = payload["question"]
            history = payload.get("history", [])
            return _run_llm_with_sources(retriever, chat_model, prompt_template, question, history)

        logger.info(
            "问答流水线已搭好（LLM 生成模式，provider=%s, system_role=%s）",
            settings.llm.provider,
            settings.llm.system_role,
        )
        return run_with_sources

    def run_fallback(payload):
        question = payload["question"]
        return _run_fallback(retriever, question)

    logger.info("问答流水线已搭好（降级模式：返回检索片段原文）")
    return run_fallback


# 造一个给 /chat 接口用的处理器，统一返回 dict。
#
# 参数：
#     settings: 全局配置。
#
# 返回：
#     可调用对象 handler(question, history) -> dict，
#     返回 {"answer": 文字, "sources": 列表}。
def make_chat_handler(settings: Settings):
    pipeline = build_rag_pipeline(settings)

    def invoke(question, history):
        payload = {"question": question, "history": history}
        # LLM 模式和降级模式都返回 dict
        result = pipeline(payload)
        if isinstance(result, dict):
            return result
        return {"answer": result, "sources": []}

    return invoke


# 流式问答：一块一块地产出答案，结束的时候给元信息。
#
# 参数：
#     settings: 全局配置。
#     question: 用户当前问题。
#     history: 历史对话列表。
#
# Yields:
#     事件字典：
#       - {"type": "sources", "sources": [...]}  检索命中的来源
#       - {"type": "delta", "content": "..."}     生成的一小段
#       - {"type": "done", "tokens": {...}}       结束，带 token 用量
#
# 注意：
#     没配 LLM 或没命中的时候，也会产出 done 事件（这时 delta 就是完整答案）。
def stream_chat(settings: Settings, question, history):
    retriever = make_retriever(settings)

    # 闲聊类问题直接给预设的友好回复，不检索
    intent, greeting_reply = classify_intent(question)
    if intent == "chat" and greeting_reply is not None:
        yield {"type": "sources", "sources": []}
        yield {"type": "delta", "content": greeting_reply}
        yield {"type": "done", "sources": [], "tokens": _zero_token_usage()}
        return

    # 检索
    doc_list = retriever.invoke(question)
    # 普通用户问答：不回传原文（with_content=False），只给来源文件名和分数
    source_list = build_source_list(doc_list, with_content=False)

    yield {"type": "sources", "sources": source_list}

    # 降级模式：直接返回原文，不调 LLM
    if not settings.llm.available:
        answer_text = _build_fallback_answer(doc_list)
        yield {"type": "delta", "content": answer_text}
        yield {"type": "done", "sources": source_list, "tokens": _zero_token_usage()}
        return

    # LLM 流式生成（不管有没有命中都过一遍 LLM，让助手角色自然回复）
    # 最多重试 1 次，避免网络抖动导致一次性失败
    chat_model = make_chat_model(settings.llm, streaming=True)
    prompt_template = build_prompt(settings.llm.system_role)
    context_text = join_docs_as_context(doc_list)  # 没命中的时候是 "(无相关知识片段)"
    history_text = join_history_as_text(history)
    messages = prompt_template.format_messages(
        context=context_text, history=history_text, question=question
    )

    max_retries = 1
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            prompt_tokens = 0
            completion_tokens = 0
            for chunk in chat_model.stream(messages):
                prompt_tokens, completion_tokens = _extract_token_usage(chunk, prompt_tokens, completion_tokens)
                # content 可能是文字或列表，这里只取文字
                content = ""
                if hasattr(chunk, "content"):
                    content = chunk.content
                else:
                    content = str(chunk)
                if content:
                    yield {"type": "delta", "content": content}

            yield {
                "type": "done",
                "sources": source_list,
                "tokens": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            return  # 成功了就直接结束，不走下面的重试
        except Exception as err:
            last_error = err
            logger.warning(
                "LLM 流式调用失败（第 %d/%d 次重试）：%s",
                attempt, max_retries, err,
            )
            if attempt < max_retries:
                # 等一秒再重试，给网络恢复的时间
                time.sleep(1)

    # 重试次数用完了还是失败，抛出异常让上层处理
    raise RuntimeError(f"LLM 调用失败（已重试 {max_retries} 次）：{last_error}")
