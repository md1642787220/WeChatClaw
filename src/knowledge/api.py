"""知识库管理接口：文档上传、分片、向量化入库与配置管理。

流程：上传文件夹 -> 分片 -> embedding 向量化 -> 写入 Chroma 向量库。
本模块只负责 HTTP 接口层（请求解析、鉴权、响应组装），
具体的分片/向量化/入库逻辑委托给 src.rag.core 与 src.rag.pipeline。

Author: MADENG
Reviewer: Li Rongdong
"""
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.auth import require_admin
from src.config import read_settings
from src.rag.core.embedder import make_embedder
from src.rag.core.splitter import split_text_into_chunks
from src.rag.core.store import add_chunks_to_index, vector_store_exists
from src.rag.pipeline.context import build_source_list
from src.rag.retriever import clear_retriever_cache, make_retriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb", tags=["knowledge-base"])

# 允许的文档类型
_ALLOWED_SUFFIXES = {".md", ".txt", ".markdown"}

# 分片结果输出目录
_OUTPUT_DIR = Path("data/chunks")


# 单个分片。
#
# 属性：
#     index: 分片在所属文件里的序号（从 0 开始）。
#     content: 分片文字。
#     metadata: 分片元数据（含 source、h1/h2/h3 标题层级等）。
class ChunkItem(BaseModel):
    index: int
    content: str
    metadata: dict


# 单个文件的分片结果。
#
# 属性：
#     filename: 原始文件名。
#     chunk_count: 分片数量。
#     chunks: 分片列表。
class SplitFileResult(BaseModel):
    filename: str
    chunk_count: int
    chunks: list


# 分片接口的整体返回。
#
# 属性：
#     files: 各文件的分片结果。
#     total_chunks: 所有文件分片总数。
#     output_dir: 如果指定了 save=true 落盘就返回落盘目录，否则 None。
class SplitResponse(BaseModel):
    files: list
    total_chunks: int
    output_dir: str = None


# 入库接口返回。
#
# 属性：
#     indexed: 实际新增的块数。
#     skipped: 因为 ID 重复而跳过的块数。
#     total_chunks: 本次解析出的总块数。
#     collection: 写入的集合名。
class IndexResponse(BaseModel):
    indexed: int
    skipped: int
    total_chunks: int
    collection: str


# 管理员检索预览返回：含原文明文（仅管理员可见）。
#
# 属性：
#     query: 原始查询词。
#     sources: 命中的来源列表（含 content 字段）。
class SearchResponse(BaseModel):
    query: str
    sources: list


# ===================== 配置管理 =====================
# 设计：
# - GET  /kb/config  返回当前生效配置（API Key / Token 字段脱敏成掩码）
# - PATCH /kb/config 更新配置项并写入 .env（不重启进程；下个请求会重新 load_settings）
# - 仅白名单字段可被前端修改，避免攻击面


# 当前生效配置（敏感字段已脱敏）。
#
# 属性：
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


# 前端可以提交的更新项。
#
# 注意：
#     所有字段都是可选（None），只提交非空字段。空字符串保留原值。
#     白名单校验在 update_config 里按 _ENV_KEYS 完成。
class ConfigUpdate(BaseModel):
    llm_provider: str = None
    llm_api_key: str = None
    llm_base_url: str = None
    llm_model: str = None
    llm_temperature: float = None
    embedding_model: str = None
    embedding_device: str = None
    retrieval_top_k: int = None
    retrieval_threshold: float = None
    vector_collection: str = None
    vector_persist_dir: str = None
    admin_token: str = None


# .env 持久化的字段白名单（field_name -> env_key）
_ENV_KEYS = {
    "llm_provider": "LLM__PROVIDER",
    "llm_api_key": "LLM_API_KEY",          # 旧式单层键，保持 .env 可读性
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


# 判断文件名后缀在不在白名单内。
#
# 参数：
#     filename: 文件名（含后缀）。
#
# 返回：
#     True 表示允许处理。
def _is_allowed(filename):
    suffix = Path(filename).suffix.lower()
    return suffix in _ALLOWED_SUFFIXES


# 把分片列表（dict）转成 ChunkItem 列表。
#
# 参数：
#     chunk_dicts: 原始分片字典列表。
#
# 返回：
#     ChunkItem 列表（带 index 序号）。
def _to_chunk_items(chunk_dicts):
    items = []
    for index, one_chunk in enumerate(chunk_dicts):
        item = ChunkItem(
            index=index,
            content=one_chunk["content"],
            metadata=one_chunk["metadata"],
        )
        items.append(item)
    return items


# 对上传文件的原始字节做分片（UTF-8 路径）。
def _split_bytes(filename, raw):
    text = raw.decode("utf-8")
    chunk_dicts = split_text_into_chunks(text, source=filename)
    items = _to_chunk_items(chunk_dicts)
    return SplitFileResult(filename=filename, chunk_count=len(items), chunks=items)


# 对上传文件的原始字节做分片（UTF-8 失败时回退 GBK）。
def _split_bytes_with_fallback(filename, raw):
    text = raw.decode("gbk", errors="replace")
    chunk_dicts = split_text_into_chunks(text, source=filename)
    items = _to_chunk_items(chunk_dicts)
    return SplitFileResult(filename=filename, chunk_count=len(items), chunks=items)


# 把一个分片结果写到磁盘上的 markdown 文件里。
#
# 参数：
#     file_result: 单个文件的分片结果。
#     out_path: 输出 markdown 路径。
def _write_split_to_file(file_result, out_path):
    lines = []
    lines.append("# 分片结果：" + file_result.filename)
    lines.append("")
    for chunk_item in file_result.chunks:
        lines.append("## 分片 " + str(chunk_item.index))
        lines.append("")
        lines.append(chunk_item.content)
        lines.append("")
    final_text = "\n".join(lines)
    out_path.write_text(final_text, encoding="utf-8")


# 上传一个或多个文档，返回分片结果（不落库、不向量化）。
#
# 参数：
#     files: 上传的文件列表。
#     _: FastAPI 依赖：管理员鉴权。
#
# 返回：
#     各文件分片结果和总数。
#
# 注意：
#     需要管理员令牌（X-Admin-Token）。
@router.post("/split", response_model=SplitResponse)
async def split_uploaded(files, _=Depends(require_admin)):
    results = []
    total_chunks = 0
    for upload_file in files:
        filename = upload_file.filename or "unknown.txt"
        if not _is_allowed(filename):
            logger.info("跳过不支持的文件：%s", filename)
            continue
        raw = await upload_file.read()
        # 空文件直接跳过
        if not raw.strip():
            logger.info("跳过空文件：%s", filename)
            continue
        # 尝试 UTF-8；失败则 GBK 兜底
        try:
            file_result = _split_bytes(filename, raw)
        except UnicodeDecodeError:
            file_result = _split_bytes_with_fallback(filename, raw)
        results.append(file_result)
        total_chunks = total_chunks + file_result.chunk_count
    return SplitResponse(files=results, total_chunks=total_chunks)


# 上传并分片，可选落盘。
#
# 参数：
#     files: 上传的文件列表。
#     save: 是否把分片结果落盘（默认 False，避免原文明文写到服务器）。
#     _: FastAPI 依赖：管理员鉴权。
#
# 返回：
#     各文件分片结果和总数；save=true 时附带 output_dir。
#
# 注意：
#     需要管理员令牌（X-Admin-Token）。
@router.post("/split-to-file", response_model=SplitResponse)
async def split_to_file(files, save=False, _=Depends(require_admin)):
    response = await split_uploaded(files)
    if save:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for file_result in response.files:
            # 用原文件名（去掉目录部分），空白字符替换为下划线
            safe_name = Path(file_result.filename).name.replace(" ", "_")
            out_path = _OUTPUT_DIR / (safe_name + ".chunks.md")
            _write_split_to_file(file_result, out_path)
            logger.info("分片已输出：%s", out_path)
        response.output_dir = str(_OUTPUT_DIR.resolve())
    else:
        response.output_dir = None
    return response


# 上传文档 -> 分片 -> 向量化 -> 写入 Chroma 向量库。
#
# 参数：
#     files: 上传的文件列表。
#     _: FastAPI 依赖：管理员鉴权。
#
# 返回：
#     入库结果（新增/跳过/总数/集合名）。
#
# 注意：
#     重复内容按固定 ID 去重；入库成功后会让检索器缓存失效。
#     需要管理员令牌（X-Admin-Token）。
@router.post("/index", response_model=IndexResponse)
async def index_uploaded(files, _=Depends(require_admin)):
    settings = read_settings()

    # 1. 分片
    split_response = await split_uploaded(files)
    documents = []
    for file_result in split_response.files:
        for chunk_item in file_result.chunks:
            doc = Document(page_content=chunk_item.content, metadata=chunk_item.metadata)
            documents.append(doc)

    if not documents:
        raise HTTPException(status_code=400, detail="未解析到可入库的文档块")

    # 2. 造向量化器并增量入库
    embedder = make_embedder(
        model_name=settings.embedding.model,
        device=settings.embedding.device,
    )
    added_count, skipped_count = add_chunks_to_index(
        documents,
        embedder,
        settings.vector_db.collection,
        settings.vector_db.persist_dir,
    )

    logger.info("入库完成 | 新增=%d 跳过=%d", added_count, skipped_count)
    # 入库后让检索器缓存失效，下次检索会加载新数据
    clear_retriever_cache()
    return IndexResponse(
        indexed=added_count,
        skipped=skipped_count,
        total_chunks=split_response.total_chunks,
        collection=settings.vector_db.collection,
    )


# 管理员检索预览：返回命中的知识片段（含原文明文 + 相关度）。
#
# 参数：
#     query: 检索查询词。
#     _: FastAPI 依赖：管理员鉴权。
#
# 返回：
#     命中来源列表（带原文明文）。
#
# 注意：
#     跟普通用户 /chat 不一样，这个接口会回传原文明文，
#     只给管理员在后台核查检索质量用。需要管理员令牌（X-Admin-Token）。
@router.get("/search", response_model=SearchResponse)
async def search_preview(
    query=Query(..., min_length=1, description="检索查询词"),
    _=Depends(require_admin),
):
    settings = read_settings()
    retriever = make_retriever(settings)
    doc_list = retriever.invoke(query)
    source_list = build_source_list(doc_list, with_content=True)
    return SearchResponse(query=query, sources=source_list)


# 对敏感字符串做脱敏：保留前 2 后 2，中间替换为 *。
#
# 参数：
#     value: 原始字符串。
#
# 返回：
#     脱敏后的字符串（短串全替换为 *）。
def _mask_secret(value):
    if not value:
        return ""
    length = len(value)
    if length <= 4:
        result = ""
        for _ in range(length):
            result = result + "*"
        return result
    middle_len = length - 4
    middle_stars = ""
    for _ in range(middle_len):
        middle_stars = middle_stars + "*"
    return value[:2] + middle_stars + value[-2:]


# 返回当前生效配置（敏感字段已脱敏为掩码）。
#
# 参数：
#     _: FastAPI 依赖：管理员鉴权。
#
# 返回：
#     当前生效配置，敏感字段已脱敏。
@router.get("/config", response_model=ConfigResponse)
async def get_config(_=Depends(require_admin)):
    s = read_settings()
    env_path = Path(".env").resolve()

    llm_section = {
        "provider": s.llm.provider,
        "apiKey": _mask_secret(s.llm.api_key),
        "apiKeySet": bool(s.llm.api_key),
        "baseUrl": s.llm.base_url,
        "model": s.llm.model,
        "temperature": s.llm.temperature,
    }
    embedding_section = {
        "model": s.embedding.model,
        "device": s.embedding.device,
        "dimension": s.embedding.dimension,
    }
    retrieval_section = {
        "topK": s.retrieval.top_k,
        "threshold": s.retrieval.threshold,
    }
    vector_db_section = {
        "collection": s.vector_db.collection,
        "persistDir": s.vector_db.persist_dir,
    }
    admin_section = {
        "tokenSet": bool(s.admin.token),
        "tokenMask": _mask_secret(s.admin.token),
    }
    return ConfigResponse(
        llm=llm_section,
        embedding=embedding_section,
        retrieval=retrieval_section,
        vector_db=vector_db_section,
        admin=admin_section,
        envPath=str(env_path),
    )


# 读 .env 已有键值。
#
# 参数：
#     path: .env 文件路径。
#
# 返回：
#     键值字典；文件不存在时返回空字典。
def _read_env_file(path):
    if not path.exists():
        return {}
    result = {}
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        partition_index = stripped.index("=")
        key = stripped[:partition_index].strip()
        value = stripped[partition_index + 1:].strip()
        # 去掉首尾成对引号
        value = value.strip("'\"")
        result[key] = value
    return result


# 把字典写回 .env（覆盖同名键，保留注释和空行）。
#
# 参数：
#     path: .env 文件路径。
#     kv: 待写入的键值。
def _write_env_file(path, kv):
    if path.exists():
        existing_lines = path.read_text(encoding="utf-8").splitlines()
    else:
        existing_lines = []

    new_lines = []
    seen_keys = set()
    for line in existing_lines:
        stripped = line.strip()
        # 形如 KEY=VALUE 的行：如果 KEY 在待写集合里就覆盖
        if stripped and not stripped.startswith("#") and "=" in stripped:
            partition_index = stripped.index("=")
            key = stripped[:partition_index].strip()
            if key in kv:
                new_lines.append(key + "=" + kv[key])
                seen_keys.add(key)
                continue
        # 注释 / 空行 / 未匹配键：原样保留
        new_lines.append(line)

    # 追加新键
    for key, value in kv.items():
        if key not in seen_keys:
            new_lines.append(key + "=" + value)

    final_text = "\n".join(new_lines) + "\n"
    path.write_text(final_text, encoding="utf-8")


# 更新配置并持久化到 .env。
#
# 参数：
#     payload: 前端提交的更新项。
#     _: FastAPI 依赖：管理员鉴权。
#
# 返回：
#     更新后的当前生效配置（跟 GET /kb/config 一致）。
#
# 注意：
#     - 只写白名单字段，避免攻击面。
#     - 写完 .env 后下个请求 load_settings() 会自动重读，不用重启进程。
@router.patch("/config", response_model=ConfigResponse)
async def update_config(payload: ConfigUpdate, _=Depends(require_admin)):
    # 提取非空字段
    raw_data = payload.model_dump(exclude_none=True)
    if not raw_data:
        raise HTTPException(status_code=400, detail="请求体为空")

    env_path = Path(".env")
    existing = _read_env_file(env_path)
    updates = {}

    for field_name, field_value in raw_data.items():
        env_key = _ENV_KEYS.get(field_name)
        if not env_key:
            continue
        value_str = str(field_value)
        existing[env_key] = value_str
        updates[env_key] = value_str

    if updates:
        _write_env_file(env_path, existing)
        logger.info("配置已更新并写入 .env：%s", list(updates.keys()))

    return await get_config()
