"""FastAPI 应用入口：企业内部知识库问答机器人。"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.chat.session import SessionManager
from src.compliance.filter import check_compliance, desensitize
from src.config import Settings, load_settings
from src.knowledge.api import router as kb_router
from src.logging_conf import setup_logging
from src.rag.engine import stream_chat

logger = logging.getLogger(__name__)

settings: Settings = load_settings()
session_manager = SessionManager(max_rounds=settings.compliance.max_history_rounds)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.app.log_level)
    logger.info("应用启动 | env=%s | llm_available=%s", settings.app.env, settings.llm.available)
    yield
    logger.info("应用关闭")


app = FastAPI(title=settings.app.name, version="0.1.0", lifespan=lifespan)

# 知识库管理接口（上传 + 分片）
app.include_router(kb_router)


@app.get("/healthz")
async def healthz() -> dict:
    """健康检查。"""
    return {"status": "ok", "env": settings.app.env}


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """问答接口（SSE 流式）：合规检查 → 流式生成 → 脱敏，支持前端中断。

    事件类型：
      - blocked：命中敏感词，直接结束
      - meta：会话信息（session_id、累计 tokens）
      - sources：检索命中的来源
      - delta：生成内容片段
      - done：本轮结束（含本轮 tokens）
      - error：处理出错
    """
    # 1. 合规检查
    ok, hit_word = check_compliance(req.question, settings.compliance.sensitive_words)
    if not ok:
        logger.warning("命中敏感词 | word=%s", hit_word)

        async def _blocked_stream() -> AsyncIterator[str]:
            yield _sse({"type": "blocked", "answer": "抱歉，您的问题涉及敏感信息，暂时无法回答。"})

        return StreamingResponse(_blocked_stream(), media_type="text/event-stream")

    session_id = req.session_id or session_manager.create()

    async def _stream() -> AsyncIterator[str]:
        # 先记录用户提问
        session_manager.append(session_id, "user", req.question)

        # 发送 meta（会话 id + 当前累计 token）
        yield _sse(
            {"type": "meta", "session_id": session_id, "tokens": session_manager.get_tokens(session_id)}
        )

        history = session_manager.get_history(session_id)
        full_answer: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        try:
            for event in stream_chat(settings, req.question, history):
                etype = event.get("type")
                if etype == "sources":
                    yield _sse({"type": "sources", "sources": event.get("sources", [])})
                elif etype == "delta":
                    piece = event.get("content", "")
                    full_answer.append(piece)
                    # 逐块脱敏（仅对当前块做，避免跨块边界问题）
                    yield _sse({"type": "delta", "content": desensitize(piece)})
                elif etype == "done":
                    prompt_tokens = event.get("tokens", {}).get("prompt_tokens", 0)
                    completion_tokens = event.get("tokens", {}).get("completion_tokens", 0)
        except Exception as e:  # noqa: BLE001
            logger.exception("流式生成失败")
            yield _sse({"type": "error", "message": "生成失败，请重试。"})
            return

        # 累计 token
        session_manager.add_tokens(
            session_id, prompt=prompt_tokens, completion=completion_tokens
        )
        # 记录助手回答
        session_manager.append(session_id, "assistant", "".join(full_answer))

        yield _sse(
            {
                "type": "done",
                "tokens": session_manager.get_tokens(session_id),
                "round_tokens": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )

    return StreamingResponse(_stream(), media_type="text/event-stream")


def _sse(data: dict[str, Any]) -> str:
    """将 dict 序列化为 SSE 事件字符串。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# 网页前端静态资源
app.mount("/static", StaticFiles(directory="src/web"), name="static")


@app.get("/")
async def index() -> FileResponse:
    """返回聊天页面。"""
    return FileResponse("src/web/index.html")


_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
    '<rect width="64" height="64" rx="12" fill="#4f6ef7"/>'
    '<text x="50%" y="56%" font-size="38" text-anchor="middle" '
    'dominant-baseline="middle" fill="white" font-family="sans-serif">'
    "K</text></svg>"
)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """内联 SVG favicon，避免 404。"""
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml")
