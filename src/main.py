"""FastAPI 应用入口：对外智能客服系统骨架。"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config import Settings, load_settings
from src.logging_conf import setup_logging

logger = logging.getLogger(__name__)

settings: Settings = load_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.app.log_level)
    logger.info("应用启动 | env=%s", settings.app.env)
    yield
    logger.info("应用关闭")


app = FastAPI(title=settings.app.name, version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    """健康检查：用于探活与部署就绪判断。"""
    return {"status": "ok", "env": settings.app.env}
