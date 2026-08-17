"""索引构建脚本：将 data/docs 下的文档切分、向量化并入库。

用法：
    uv run python -m scripts.build_index

Author: MADENG
Reviewer: Li Rongdong
"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

from src.config import load_settings
from src.rag.core.embedder import build_embeddings
from src.rag.core.loader import load_text
from src.rag.core.splitter import split_markdown
from src.rag.core.store import build_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DOCS_DIR = Path("data/docs")


# 遍历目录下所有 md / txt 文件，切分为 Document。
#
# Args:
#     docs_dir: 文档根目录。
#
# Returns:
#     已切分的 Document 列表；目录不存在时返回空列表。
def collect_docs(docs_dir: Path) -> list[Document]:
    if not docs_dir.exists():
        logger.warning("文档目录不存在：%s", docs_dir)
        return []

    documents: list[Document] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        logger.info("处理文档：%s", path)
        text = load_text(path)
        for chunk in split_markdown(text, source=str(path)):
            documents.append(
                Document(page_content=chunk["content"], metadata=chunk["metadata"])
            )
    return documents


# 脚本主入口。
def main() -> None:
    settings = load_settings()
    documents = collect_docs(DOCS_DIR)

    if not documents:
        logger.warning("未找到可入库的文档，请在 data/docs 下放置 .md 或 .txt 文件。")
        return

    embeddings = build_embeddings(
        model_name=settings.embedding.model,
        device=settings.embedding.device,
    )
    store = build_index(
        docs=documents,
        embeddings=embeddings,
        collection_name=settings.vector_db.collection,
        persist_dir=settings.vector_db.persist_dir,
    )
    logger.info("索引构建完成，共 %d 个文档块，集合：%s", len(documents), store._collection.name)


if __name__ == "__main__":
    main()
