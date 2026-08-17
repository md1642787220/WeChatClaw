"""知识库管理接口：文档上传、分片、向量化入库与配置管理。

流程：上传文件夹 → 分片 → embedding 向量化 → 写入 Chroma 向量库。
本模块只负责 HTTP 接口层（请求解析、鉴权、响应组装），
具体的分片/向量化/入库逻辑委托给 ``src.rag.core`` 与 ``src.rag.pipeline``。

Author: MADENG
Reviewer: Li Rongdong
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.auth import require_admin
from src.config import load_settings
from src.rag.core.embedder import build_embeddings
from src.rag.core.splitter import split_markdown
from src.rag.core.store import add_documents, load_store, store_exists
from src.rag.pipeline.context import build_source_objs
from src.rag.retriever import build_retriever, invalidate_retriever_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb", tags=["knowledge-base"])

# 允许的文档类型
_ALLOWED_SUFFIXES = {".md", ".txt", ".markdown"}

# 分片结果输出目录
_OUTPUT_DIR = Path("data/chunks")


# 单个分片。
#
# Attributes:
#     index: 分片在所属文件中的序号（从 0 开始）。
#     content: 分片文本。
#     metadata: 分片元数据（含 source、h1/h2/h3 标题层级等）。
class ChunkItem(BaseModel):
    index: int
    content: str
    metadata: dict


# 单个文件的分片结果。
#
# Attributes:
#     filename: 原始文件名。
#     chunk_count: 分片数量。
#     chunks: 分片列表。
class SplitFileResult(BaseModel):
    filename: str
    chunk_count: int
    chunks: list[ChunkItem]


# 分片接口整体返回。
#
# Attributes:
#     files: 各文件的分片结果。
#     total_chunks: 所有文件分片总数。
#     output_dir: 若指定 ``save=true`` 落盘则为落盘目录，否则 None。
class SplitResponse(BaseModel):
    files: list[SplitFileResult]
    total_chunks: int
    output_dir: str | None = None


# 判断文件名后缀是否在白名单内。
#
# Args:
#     filename: 文件名（含后缀）。
#
# Returns:
#     True 表示允许处理。
def _is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in _ALLOWED_SUFFIXES


# 对上传文件的原始字节做分片。
#
# Args:
#     filename: 文件名，用作分片的 source 标识。
#     raw: 文件的原始字节内容。
#
# Returns:
#     单文件分片结果。
#
# Notes:
#     尝试按 UTF-8 解码；失败时按 GBK 兜底（兼容常见中文编码）。
def _split_bytes(filename: str, raw: bytes) -> SplitFileResult:
    text = raw.decode("utf-8")
    chunks = split_markdown(text, source=filename)
    items = [
        ChunkItem(index=i, content=c["content"], metadata=c["metadata"])
        for i, c in enumerate(chunks)
    ]
    return SplitFileResult(filename=filename, chunk_count=len(items), chunks=items)


# 上传一个或多个文档，返回分片结果（不落库、不向量化）。
#
# Args:
#     files: 上传的文件列表（前端用 ``<input type="file" webkitdirectory>`` 选择整个文件夹）。
#     _: FastAPI 依赖：管理员鉴权。
#
# Returns:
#     各文件分片结果与总数。
#
# Notes:
#     需要管理员令牌（X-Admin-Token）。
@router.post("/split", response_model=SplitResponse)
async def split_uploaded(
    files: list[UploadFile], _: None = Depends(require_admin)
) -> SplitResponse:
    results: list[SplitFileResult] = []
    total = 0
    for f in files:
        filename = f.filename or "unknown.txt"
        if not _is_allowed(filename):
            logger.info("跳过不支持的文件：%s", filename)
            continue
        raw = await f.read()
        if not raw.strip():
            logger.info("跳过空文件：%s", filename)
            continue
        try:
            res = _split_bytes(filename, raw)
        except UnicodeDecodeError:
            # UTF-8 失败，尝试 GBK
            text = raw.decode("gbk", errors="replace")
            chunks = split_markdown(text, source=filename)
            items = [
                ChunkItem(index=i, content=c["content"], metadata=c["metadata"])
                for i, c in enumerate(chunks)
            ]
            res = SplitFileResult(filename=filename, chunk_count=len(items), chunks=items)
        results.append(res)
        total += res.chunk_count

    return SplitResponse(files=results, total_chunks=total)


# 上传并分片，可选落盘。
#
# Args:
#     files: 上传的文件列表。
#     save: 是否将分片结果落盘（默认 False，避免原文明文写入服务器磁盘）。
#     _: FastAPI 依赖：管理员鉴权。
#
# Returns:
#     各文件分片结果与总数；当 ``save=true`` 时附带 ``output_dir``。
#
# Notes:
#     需要管理员令牌（X-Admin-Token）。
@router.post("/split-to-file", response_model=SplitResponse)
async def split_to_file(
    files: list[UploadFile],
    save: bool = False,
    _: None = Depends(require_admin),
) -> SplitResponse:
    resp = await split_uploaded(files)

    if save:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for fr in resp.files:
            safe_name = Path(fr.filename).name.replace(" ", "_")
            out_path = _OUTPUT_DIR / f"{safe_name}.chunks.md"
            lines = [f"# 分片结果：{fr.filename}", ""]
            for c in fr.chunks:
                lines.append(f"## 分片 {c.index}")
                lines.append("")
                lines.append(c.content)
                lines.append("")
            out_path.write_text("\n".join(lines), encoding="utf-8")
            logger.info("分片已输出：%s", out_path)
        resp.output_dir = str(_OUTPUT_DIR.resolve())
    else:
        resp.output_dir = None  # 不落盘
    return resp


# 入库接口返回。
#
# Attributes:
#     indexed: 实际新增的块数。
#     skipped: 因稳定 ID 重复而跳过的块数。
#     total_chunks: 本次解析出的总块数。
#     collection: 写入的集合名。
class IndexResponse(BaseModel):
    indexed: int
    skipped: int
    total_chunks: int
    collection: str


# 上传文档 → 分片 → 向量化 → 写入 Chroma 向量库。
#
# Args:
#     files: 上传的文件列表。
#     _: FastAPI 依赖：管理员鉴权。
#
# Returns:
#     入库结果（新增/跳过/总数/集合名）。
#
# Notes:
#     重复内容按稳定 ID 去重；入库成功后会使检索器缓存失效。
#     需要管理员令牌（X-Admin-Token）。
@router.post("/index", response_model=IndexResponse)
async def index_uploaded(
    files: list[UploadFile], _: None = Depends(require_admin)
) -> IndexResponse:
    settings = load_settings()

    # 1. 分片
    split_resp = await split_uploaded(files)
    documents: list[Document] = []
    for fr in split_resp.files:
        for c in fr.chunks:
            documents.append(
                Document(page_content=c.content, metadata=c.metadata)
            )

    if not documents:
        raise HTTPException(status_code=400, detail="未解析到可入库的文档块")

    # 2. 构建 embedding 并增量入库
    embeddings = build_embeddings(
        model_name=settings.embedding.model,
        device=settings.embedding.device,
    )
    added, skipped = add_documents(
        docs=documents,
        embeddings=embeddings,
        collection_name=settings.vector_db.collection,
        persist_dir=settings.vector_db.persist_dir,
    )

    logger.info("入库完成 | 新增=%d 跳过=%d", added, skipped)
    # 入库后使检索器缓存失效，下次检索将加载新数据
    invalidate_retriever_cache()
    return IndexResponse(
        indexed=added,
        skipped=skipped,
        total_chunks=split_resp.total_chunks,
        collection=settings.vector_db.collection,
    )


# 管理员检索预览返回：含原文明文（仅管理员可见）。
#
# Attributes:
#     query: 原始查询词。
#     sources: 命中的来源列表（含 content 字段）。
class SearchResponse(BaseModel):
    query: str
    sources: list[dict]


# 管理员检索预览：返回命中的知识片段（含原文明文 + 相关度）。
#
# Args:
#     query: 检索查询词。
#     _: FastAPI 依赖：管理员鉴权。
#
# Returns:
#     命中来源列表（带原文明文）。
#
# Notes:
#     与普通用户 /chat 不同，本接口**会回传原文明文**，
#     仅供管理员在后台核查检索质量。需要管理员令牌（X-Admin-Token）。
@router.get("/search", response_model=SearchResponse)
async def search_preview(
    query: str = Query(..., min_length=1, description="检索查询词"),
    _: None = Depends(require_admin),
) -> SearchResponse:
    settings = load_settings()
    retriever = build_retriever(settings)
    docs = retriever.invoke(query)
    sources = build_source_objs(docs, with_content=True)
    return SearchResponse(query=query, sources=sources)


# ===================== 配置管理 =====================
# 设计：
# - GET  /kb/config  返回当前生效配置（API Key / Token 字段脱敏成掩码）
# - PATCH /kb/config 更新配置项并写入 .env（不重启进程；下个请求会重新 load_settings）
# - 仅白名单字段可被前端修改，避免攻击面

# 当前生效配置（敏感字段已脱敏）。
#
# Attributes:
#     llm: LLM 配置（apiKey 已脱敏为掩码）。
#     embedding: Embedding 配置。
#     retrieval: 检索配置（topK、threshold）。
#     vector_db: 向量库配置（collection、persistDir）。
#     admin: 管理员配置（tokenSet/tokenMask）。
#     env_path: .env 文件绝对路径（前端展示用，alias 兼容前端的 envPath）。
class ConfigResponse(BaseModel):
    model_config = {"populate_by_name": True}

    llm: dict
    embedding: dict
    retrieval: dict
    vector_db: dict
    admin: dict
    env_path: str = Field(..., alias="envPath")


# 前端可提交的更新项。
#
# Notes:
#     所有字段都是可选（None），仅提交非空字段。空字符串保留原值。
#     白名单校验在 :func:`update_config` 中按 ``_ENV_KEYS`` 完成。
class ConfigUpdate(BaseModel):
    llm_provider: Literal["deepseek", "openai", "qwen", "hunyuan", "mock"] | None = None
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_temperature: float | None = None
    embedding_model: str | None = None
    embedding_device: Literal["cpu", "cuda"] | None = None
    retrieval_top_k: int | None = None
    retrieval_threshold: float | None = None
    vector_collection: str | None = None
    vector_persist_dir: str | None = None
    admin_token: str | None = None


# 对敏感字符串做脱敏：保留前 2 后 2，中间替换为 *。
#
# Args:
#     value: 原始字符串。
#
# Returns:
#     脱敏后的字符串（短串全替换为 *）。
def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * len(value)
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


# 返回当前生效配置（敏感字段已脱敏为掩码）。
#
# Args:
#     _: FastAPI 依赖：管理员鉴权。
#
# Returns:
#     当前生效配置，敏感字段已脱敏。
@router.get("/config", response_model=ConfigResponse)
async def get_config(_: None = Depends(require_admin)) -> ConfigResponse:
    s = load_settings()
    env_path = Path(".env").resolve()
    return ConfigResponse(
        llm={
            "provider": s.llm.provider,
            "apiKey": _mask(s.llm.api_key),
            "apiKeySet": bool(s.llm.api_key),
            "baseUrl": s.llm.base_url,
            "model": s.llm.model,
            "temperature": s.llm.temperature,
        },
        embedding={
            "model": s.embedding.model,
            "device": s.embedding.device,
            "dimension": s.embedding.dimension,
        },
        retrieval={
            "topK": s.retrieval.top_k,
            "threshold": s.retrieval.threshold,
        },
        vector_db={
            "collection": s.vector_db.collection,
            "persistDir": s.vector_db.persist_dir,
        },
        admin={
            "tokenSet": bool(s.admin.token),
            "tokenMask": _mask(s.admin.token),
        },
        envPath=str(env_path),
    )


# .env 持久化的字段白名单
_ENV_KEYS: dict[str, str] = {
    "llm_provider": "LLM__PROVIDER",
    "llm_api_key": "LLM_API_KEY",      # 旧式单层键，保持 .env 可读性
    "llm_base_url": "LLM__BASE_URL",
    "llm_model": "LLM__MODEL",
    "llm_temperature": "LLM__TEMPERATURE",
    "embedding_model": "EMBEDDING__MODEL",
    "embedding_device": "EMBEDDING__DEVICE",
    "retrieval_top_k": "RETRIEVAL__TOP_K",
    "retrieval_threshold": "RETRIEVAL__THRESHOLD",
    "vector_collection": "VECTOR_DB__COLLECTION",
    "vector_persist_dir": "VECTOR_DB__PERSIST_DIR",
    "admin_token": "ADMIN_TOKEN",
}


# 读取 .env 已有键值。
#
# Args:
#     path: .env 文件路径。
#
# Returns:
#     键值字典；文件不存在时返回空字典。
def _read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        result[k.strip()] = v.strip().strip("'\"")
    return result


# 将字典写回 .env（覆盖同名键，保留注释与空行）。
#
# Args:
#     path: .env 文件路径。
#     kv: 待写入的键值。
def _write_env(path: Path, kv: dict[str, str]) -> None:
    lines: list[str] = []
    if path.exists():
        existing_lines = path.read_text(encoding="utf-8").splitlines()
    else:
        existing_lines = []
    seen: set[str] = set()
    for line in existing_lines:
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k = s.partition("=")[0].strip()
            if k in kv:
                lines.append(f"{k}={kv[k]}")
                seen.add(k)
                continue
        lines.append(line)
    # 追加新键
    for k, v in kv.items():
        if k not in seen:
            lines.append(f"{k}={v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# 更新配置并持久化到 .env。
#
# Args:
#     payload: 前端提交的更新项。
#     _: FastAPI 依赖：管理员鉴权。
#
# Returns:
#     更新后的当前生效配置（与 GET /kb/config 一致）。
#
# Notes:
#     - 仅写白名单字段，避免攻击面。
#     - 写入 .env 后下个请求 ``load_settings()`` 会自动重新加载
#       （Settings 走 BaseSettings，每次构造都重读环境变量与 .env 兜底）。无需重启进程。
@router.patch("/config", response_model=ConfigResponse)
async def update_config(
    payload: ConfigUpdate, _: None = Depends(require_admin)
) -> ConfigResponse:
    data = payload.model_dump(exclude_none=True)
    if not data:
        raise HTTPException(status_code=400, detail="请求体为空")

    env_path = Path(".env")
    existing = _read_env(env_path)
    updates: dict[str, str] = {}
    for field_name, value in data.items():
        env_key = _ENV_KEYS.get(field_name)
        if not env_key:
            continue
        existing[env_key] = str(value)
        updates[env_key] = str(value)

    if updates:
        _write_env(env_path, existing)
        logger.info("配置已更新并写入 .env：%s", list(updates.keys()))

    return await get_config()
