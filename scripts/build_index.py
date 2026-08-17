# 索引构建脚本：把 data/docs 下的文档切分、向量化后入库。
#
# 用法：
#     uv run python -m scripts.build_index
#
# Author: MADENG
# Reviewer: Li Rongdong
import logging
from pathlib import Path

from langchain_core.documents import Document

from src.config import read_settings
from src.rag.core.embedder import make_embedder
from src.rag.core.splitter import split_text_into_chunks
from src.rag.core.store import build_full_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DOCS_DIR = Path("data/docs")


# 判断文件后缀是不是需要处理的文本格式。
#
# 参数：
#     suffix: 文件后缀（已 lower）。
#
# 返回：
#     True 表示要处理。
def _is_doc_file(suffix):
    return suffix == ".md" or suffix == ".txt"


# 把一个块字典转成 Document。
def _chunk_dict_to_doc(chunk_dict):
    return Document(page_content=chunk_dict["content"], metadata=chunk_dict["metadata"])


# 遍历目录下所有 md / txt 文件，切成 Document。
#
# 参数：
#     docs_dir: 文档根目录。
#
# 返回：
#     切好的 Document 列表；目录不存在时返回空列表。
def collect_all_docs(docs_dir: Path):
    if not docs_dir.exists():
        logger.warning("文档目录不存在：%s", docs_dir)
        return []

    all_documents = []
    # 收集所有文件路径后排序，确保每次构建顺序一致
    all_paths = []
    for path in docs_dir.rglob("*"):
        if not path.is_file():
            continue
        all_paths.append(path)
    all_paths.sort()

    for path in all_paths:
        suffix = path.suffix.lower()
        if not _is_doc_file(suffix):
            continue
        logger.info("处理文档：%s", path)
        text = path.read_text(encoding="utf-8")
        chunk_dicts = split_text_into_chunks(text, source=str(path))
        for chunk_dict in chunk_dicts:
            all_documents.append(_chunk_dict_to_doc(chunk_dict))
    return all_documents


# 脚本主入口。
def main():
    settings = read_settings()
    all_documents = collect_all_docs(DOCS_DIR)

    if not all_documents:
        logger.warning("没找到可入库的文档，请在 data/docs 下放 .md 或 .txt 文件。")
        return

    embedder = make_embedder(
        model_name=settings.embedding.model,
        device=settings.embedding.device,
    )
    store = build_full_index(
        doc_list=all_documents,
        embeddings=embedder,
        collection_name=settings.vector_db.collection,
        persist_dir=settings.vector_db.persist_dir,
    )
    collection_name_value = store._collection.name
    logger.info("索引构建完成，共 %d 个文档块，集合：%s", len(all_documents), collection_name_value)


if __name__ == "__main__":
    main()
