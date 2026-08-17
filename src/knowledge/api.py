"""知识库管理接口：文档上传、分片、向量化入库。

流程：上传文件夹 → 分片 → embedding 向量化 → 写入 Chroma 向量库。
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from langchain_core.documents import Document
from pydantic import BaseModel

from src.auth import require_admin
from src.config import load_settings
from src.knowledge.embeddings import build_embeddings
from src.knowledge.loader import split_markdown
from src.knowledge.vector_store import add_documents
from src.rag.engine import _build_source_objs
from src.rag.retriever import build_retriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb", tags=["knowledge-base"])

# 允许的文档类型
_ALLOWED_SUFFIXES = {".md", ".txt", ".markdown"}

# 分片结果输出目录
_OUTPUT_DIR = Path("data/chunks")


class ChunkItem(BaseModel):
    """单个分片。"""

    index: int
    content: str
    metadata: dict


class SplitFileResult(BaseModel):
    """单个文件的分片结果。"""

    filename: str
    chunk_count: int
    chunks: list[ChunkItem]


class SplitResponse(BaseModel):
    """分片接口整体返回。"""

    files: list[SplitFileResult]
    total_chunks: int
    output_dir: str | None = None


def _is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in _ALLOWED_SUFFIXES


def _split_bytes(filename: str, raw: bytes) -> SplitFileResult:
    """对上传文件的原始字节做分片。

    尝试按 UTF-8 解码；失败时按 GBK 兜底（兼容常见中文编码）。
    """
    text = raw.decode("utf-8")
    chunks = split_markdown(text, source=filename)
    items = [
        ChunkItem(index=i, content=c["content"], metadata=c["metadata"])
        for i, c in enumerate(chunks)
    ]
    return SplitFileResult(filename=filename, chunk_count=len(items), chunks=items)


@router.post("/split", response_model=SplitResponse)
async def split_uploaded(
    files: list[UploadFile], _: None = Depends(require_admin)
) -> SplitResponse:
    """上传一个或多个文档，返回分片结果（不落库、不向量化）。

    前端通过 ``<input type="file" webkitdirectory>`` 选择整个文件夹，
    即可一次性上传目录内所有 .md/.txt 文档。

    需要管理员令牌（X-Admin-Token）。
    """
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


@router.post("/split-to-file", response_model=SplitResponse)
async def split_to_file(
    files: list[UploadFile],
    save: bool = False,
    _: None = Depends(require_admin),
) -> SplitResponse:
    """上传并分片。

    默认**不落盘**（避免原文明文写入服务器磁盘）。仅当传入 ``save=true`` 时，
    才将分片结果写入 data/chunks 目录（markdown 格式），供管理员本地核查。

    需要管理员令牌（X-Admin-Token）。
    """
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


class IndexResponse(BaseModel):
    """入库接口返回。"""

    indexed: int
    skipped: int
    total_chunks: int
    collection: str


@router.post("/index", response_model=IndexResponse)
async def index_uploaded(
    files: list[UploadFile], _: None = Depends(require_admin)
) -> IndexResponse:
    """上传文档 → 分片 → 向量化 → 写入 Chroma 向量库。

    重复内容按稳定 ID 去重，返回实际新增与跳过的块数。

    需要管理员令牌（X-Admin-Token）。
    """
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
    return IndexResponse(
        indexed=added,
        skipped=skipped,
        total_chunks=split_resp.total_chunks,
        collection=settings.vector_db.collection,
    )


class SearchResponse(BaseModel):
    """管理员检索预览返回：含原文明文（仅管理员可见）。"""

    query: str
    sources: list[dict]


@router.get("/search", response_model=SearchResponse)
async def search_preview(
    query: str = Query(..., min_length=1, description="检索查询词"),
    _: None = Depends(require_admin),
) -> SearchResponse:
    """管理员检索预览：返回命中的知识片段（含原文明文 + 相关度）。

    与普通用户 /chat 不同，本接口**会回传原文明文**，仅供管理员在后台核查
    检索质量使用。需要管理员令牌（X-Admin-Token）。
    """
    settings = load_settings()
    retriever = build_retriever(settings)
    docs = retriever.invoke(query)
    sources = _build_source_objs(docs, with_content=True)
    return SearchResponse(query=query, sources=sources)
