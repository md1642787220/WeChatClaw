"""FastAPI 应用入口：企业内部知识库问答机器人。

提供的接口：/healthz、/chat（SSE 流式）、/kb/*（知识库管理）、/（前端页面）。

Author: MADENG
Reviewer: Li Rongdong
"""
import json
import logging
from contextlib import asynccontextmanager

from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.chat.session import SessionManager
from src.compliance.filter import check_compliance, mask_sensitive_text
from src.config import read_settings
from src.knowledge.api import router as kb_router
from src.logging_conf import setup_logging
from src.rag.engine import stream_chat
from src.rag.retriever import warm_up_retriever

logger = logging.getLogger(__name__)

settings = read_settings()
session_manager = SessionManager(max_rounds=settings.compliance.max_history_rounds)


# /chat 接口的请求体。
#
# 属性：
#     question: 用户提问（1~2000 字）。
#     session_id: 可选会话 id，缺省或为 null 时由服务端自动创建。
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None


# 工具函数：把 dict 序列化成 SSE 事件字符串。
#
# 参数：
#     event_data: 事件数据。
#
# 返回：
#     符合 text/event-stream 协议的字符串（含 data: 前缀和双换行）。
def _to_sse(event_data):
    payload = json.dumps(event_data, ensure_ascii=False)
    return "data: " + payload + "\n\n"


# 异步生成器：被敏感词拦截时返回 blocked 事件后立即结束。
async def _blocked_stream():
    yield _to_sse({
        "type": "blocked",
        "answer": "抱歉，您的问题涉及敏感信息，暂时无法回答。",
    })


# 异步生成器：正常流式问答主流程。
async def _stream_chat(session_id, question, history):
    # 先记录用户提问
    session_manager.append(session_id, "user", question)

    # 发送 meta（会话 id + 当前累计 token）
    meta_payload = {
        "type": "meta",
        "session_id": session_id,
        "tokens": session_manager.get_tokens(session_id),
    }
    yield _to_sse(meta_payload)

    full_answer_parts = []
    prompt_tokens = 0
    completion_tokens = 0

    # 流式调用 RAG
    try:
        for event in stream_chat(settings, question, history):
            event_type = event.get("type")
            if event_type == "sources":
                sources_value = event.get("sources", [])
                yield _to_sse({"type": "sources", "sources": sources_value})
            elif event_type == "delta":
                piece = event.get("content", "")
                full_answer_parts.append(piece)
                # 逐块脱敏（只对当前块做，避免跨块边界问题）
                yield _to_sse({"type": "delta", "content": mask_sensitive_text(piece)})
            elif event_type == "done":
                tokens_dict = event.get("tokens", {})
                prompt_tokens = tokens_dict.get("prompt_tokens", 0)
                completion_tokens = tokens_dict.get("completion_tokens", 0)
    except Exception as error:  # noqa: BLE001
        logger.exception("流式生成失败")
        yield _to_sse({"type": "error", "message": "生成失败，请重试。"})
        return

    # 累计 token
    session_manager.add_tokens(
        session_id, prompt=prompt_tokens, completion=completion_tokens
    )
    # 记录助手回答（把片段按顺序拼成完整文本）
    assistant_answer = "".join(full_answer_parts)
    session_manager.append(session_id, "assistant", assistant_answer)

    done_payload = {
        "type": "done",
        "tokens": session_manager.get_tokens(session_id),
        "round_tokens": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    yield _to_sse(done_payload)


# FastAPI lifespan：启动时预热知识库，关闭时只记日志。
@asynccontextmanager
async def lifespan(app):
    setup_logging(settings.app.log_level)
    logger.info(
        "应用启动 | env=%s | llm_available=%s",
        settings.app.env,
        settings.llm.available,
    )
    # 启动完成后预热知识库：预加载 embedding 模型和向量库，让第一个请求直接可用
    try:
        warm_up_retriever(settings)
    except Exception as warmup_error:  # noqa: BLE001
        # 预热失败不应该阻止服务启动，降级成「首次检索时再懒加载」
        logger.warning(
            "知识库预热失败（将降级为首次检索时懒加载）：%s", warmup_error
        )
    yield
    logger.info("应用关闭")


app = FastAPI(
    title=settings.app.name,
    version="0.1.0",
    lifespan=lifespan,
)


# 把 FastAPI 请求体校验错误（422）转成可读的字符串 detail。
#
# 默认返回 {"detail": [{"type","loc","msg",...}, ...]}，前端直接拼接会显示
# "[object Object]"。这里遍历错误列表，拼成 "字段路径: 错误信息" 的字符串。
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    parts = []
    for err in exc.errors():
        loc = err.get("loc", [])
        # 去掉开头的 body/query/path 等来源标记，只保留字段名
        field = ".".join(str(p) for p in loc if p not in ("body", "query", "path"))
        msg = err.get("msg", "")
        parts.append((field + ": " + msg) if field else msg)
    message = "；".join(parts) if parts else "请求参数校验失败"
    logger.warning("请求校验失败 | path=%s | %s", request.url.path, message)
    return JSONResponse(status_code=422, content={"detail": message})


# 知识库管理接口（上传 + 分片）
app.include_router(kb_router)


# 健康检查接口。
#
# 返回：
#     服务状态和运行环境。
@app.get("/healthz")
async def healthz():
    return {"status": "ok", "env": settings.app.env}


# 问答接口（SSE 流式）。
#
# 流程：合规检查 -> 流式生成 -> 脱敏，支持前端中断。
#
# 参数：
#     req: 问答请求体。
#
# 返回：
#     SSE 流式响应。事件类型：
#       - blocked：命中敏感词，直接结束
#       - meta：会话信息（session_id、累计 tokens）
#       - sources：检索命中的来源
#       - delta：生成内容片段
#       - done：本轮结束（含本轮 tokens）
#       - error：处理出错
@app.post("/chat")
async def chat(req: ChatRequest):
    # 1. 合规检查
    is_allowed, hit_word = check_compliance(
        req.question, settings.compliance.sensitive_words
    )
    if not is_allowed:
        logger.warning("命中敏感词 | word=%s", hit_word)
        return StreamingResponse(_blocked_stream(), media_type="text/event-stream")

    # 2. 决定 session_id（前端传入则复用，否则新建）
    if req.session_id is None:
        session_id = session_manager.create()
    else:
        session_id = req.session_id

    # 3. 拉取历史，启动流式生成
    history = session_manager.get_history(session_id)
    generator = _stream_chat(session_id, req.question, history)
    return StreamingResponse(generator, media_type="text/event-stream")


# 网页前端静态资源
app.mount("/static", StaticFiles(directory="src/web"), name="static")


# 返回聊天页面。
@app.get("/")
async def index():
    return FileResponse("src/web/index.html")


# 内联 SVG favicon，避免 404
_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
    '<rect width="64" height="64" rx="12" fill="#4f6ef7"/>'
    '<text x="50%" y="56%" font-size="38" text-anchor="middle" '
    'dominant-baseline="middle" fill="white" font-family="sans-serif">'
    "K</text></svg>"
)


# 内联 SVG favicon，避免 404。
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml")
