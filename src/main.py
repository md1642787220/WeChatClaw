"""FastAPI 应用入口：企业内部知识库问答机器人。"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.chat.session import SessionManager
from src.compliance.filter import check_compliance, desensitize
from src.config import Settings, load_settings
from src.logging_conf import setup_logging
from src.rag.engine import build_chat_handler

logger = logging.getLogger(__name__)

settings: Settings = load_settings()
session_manager = SessionManager(max_rounds=settings.compliance.max_history_rounds)
chat_handler = build_chat_handler(settings)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []
    session_id: str | None = None
    blocked: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.app.log_level)
    logger.info("应用启动 | env=%s | llm_available=%s", settings.app.env, settings.llm.available)
    yield
    logger.info("应用关闭")


app = FastAPI(title=settings.app.name, version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    """健康检查。"""
    return {"status": "ok", "env": settings.app.env}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """问答接口：合规检查 → 检索生成 → 脱敏返回。"""
    # 1. 合规检查
    ok, hit_word = check_compliance(req.question, settings.compliance.sensitive_words)
    if not ok:
        logger.warning("命中敏感词 | word=%s", hit_word)
        return ChatResponse(
            answer="抱歉，您的问题涉及敏感信息，暂时无法回答。",
            blocked=True,
            session_id=req.session_id,
        )

    # 2. 会话上下文
    session_id = req.session_id or session_manager.create()
    history = session_manager.get_history(session_id)

    # 3. RAG 生成
    try:
        result = chat_handler(req.question, history)
    except Exception as e:  # noqa: BLE001
        logger.exception("问答处理失败")
        raise HTTPException(status_code=500, detail="内部处理错误") from e

    answer = desensitize(result.get("answer", ""))

    # 4. 记录本轮对话
    session_manager.append(session_id, "user", req.question)
    session_manager.append(session_id, "assistant", answer)

    return ChatResponse(
        answer=answer,
        sources=result.get("sources", []),
        session_id=session_id,
    )


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
