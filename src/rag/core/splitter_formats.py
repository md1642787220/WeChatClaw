"""多格式文档分片模块：按文件格式走不同的分片策略。

这个模块负责「识别文件格式 -> 选对应分片函数 -> 产出统一格式的分片」。
跟 splitter.py 的分工：

- splitter.py        只处理纯文本/Markdown，按标题层级 + 长度切分。
- splitter_formats.py 面向多格式（txt/md/pdf/docx/xlsx/csv/json/xml/html），
                      每种格式一个独立的分片函数，签名统一，方便扩展。

统一约定：
- 所有分片函数签名一致：split_<fmt>(data: bytes, source: str, **kwargs) -> list[dict]
    - data    原始字节（这样能处理二进制格式，比如 pdf/docx/xlsx）。
    - source  来源文件名，会写进每个块的 metadata["source"]。
    - kwargs  预留的扩展参数（如 chunk_size、overlap），各格式按需消费。
- 返回值统一是块列表，每个元素是 {"content": str, "metadata": dict}，
  跟 splitter.split_text_into_chunks 的返回结构一致，下游无感衔接。
- 每个函数内部：先「提取文本」，再「复用 splitter 切分」，最后统一归一化。

Author: MADENG
Reviewer: Li Rongdong
"""
from pathlib import Path

from src.rag.core import splitter


# ======================================================================
# 通用工具
# ======================================================================


# 把 bytes 解码成文字：先 UTF-8，失败回退 GBK。
def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("gbk", errors="replace")


# 提取文本 -> 复用 splitter 切分 -> 归一化 metadata，产出统一分片。
#
# 参数：
#     text: 已提取出的纯文本。
#     source: 来源文件名。
#
# 返回：
#     统一格式的块列表。
def _split_text_like(text: str, source: str):
    chunks = splitter.split_text_into_chunks(text, source=source)
    return _normalize_chunks(chunks, source)


# 归一化：保证每个块都有 content / metadata 且 metadata 带 source 与 fmt。
#
# 参数：
#     chunks: 上游产出的块列表（dict 或 Document 混合均兼容）。
#     source: 来源文件名。
#     fmt: 格式标识（如 "pdf"/"docx"），写入 metadata["fmt"] 方便溯源。
def _normalize_chunks(chunks, source: str, fmt: str):
    result = []
    for one in chunks:
        if hasattr(one, "page_content"):
            content = one.page_content
            meta = dict(one.metadata)
        elif isinstance(one, dict):
            content = one.get("content", "")
            meta = dict(one.get("metadata", {}))
        else:
            continue
        meta.setdefault("source", source)
        meta["fmt"] = fmt
        result.append({"content": content, "metadata": meta})
    return result


# ======================================================================
# 各格式的分片函数（签名统一：split_<fmt>(data, source, **kwargs)）
# ======================================================================


# 纯文本 / Markdown 分片。
#
# 说明：txt 与 md 直接按文本处理；md 复用 splitter 的标题层级切分。
def split_txt(data: bytes, source: str, **kwargs):
    text = _decode(data)
    return _split_text_like(text, source)


def split_md(data: bytes, source: str, **kwargs):
    return split_txt(data, source, **kwargs)


# PDF 分片。
#
# 说明：先提取每一页文字，再交给 splitter 切分。
# 依赖：pypdf（或 pdfplumber）。当前未实现，留占位。
def split_pdf(data: bytes, source: str, **kwargs):
    # TODO: 接入 PDF 文本提取库（如 pypdf.PdfReader），逐页提取文本后
    #       调用 _split_text_like 切分，并在 metadata 记录页码。
    raise NotImplementedError("PDF 分片尚未实现，请先安装 pypdf 并补充提取逻辑")


# Word 文档分片。
#
# 说明：解析 docx 的段落与表格，拼接成文本后切分。
# 依赖：python-docx。当前未实现，留占位。
def split_docx(data: bytes, source: str, **kwargs):
    # TODO: 接入 python-docx，遍历 document.paragraphs 与 document.tables
    #       提取文本后调用 _split_text_like 切分。
    raise NotImplementedError("DOCX 分片尚未实现，请先安装 python-docx 并补充提取逻辑")


# Excel 表格分片。
#
# 说明：按「工作表 -> 行」展开成表格化的文本，再切分。
# 依赖：openpyxl。当前未实现，留占位。
def split_xlsx(data: bytes, source: str, **kwargs):
    # TODO: 接入 openpyxl，遍历各 sheet 的 rows，把每行拼成
    #       "列名: 值" 的文本，再调用 _split_text_like 切分。
    raise NotImplementedError("XLSX 分片尚未实现，请先安装 openpyxl 并补充提取逻辑")


# CSV 分片。
#
# 说明：CSV 本质是文本，但需要按行结构化。当前先按纯文本切分，
#       后续可改为「按表头 + 行」展开。
def split_csv(data: bytes, source: str, **kwargs):
    text = _decode(data)
    # TODO: 可用 csv 模块解析，把每行拼成 "表头: 值" 再切分，保留更细的语义。
    return _split_text_like(text, source)


# JSON 分片。
#
# 说明：JSON 结构化数据，理想是按字段/记录切。当前先按格式化文本切分。
def split_json(data: bytes, source: str, **kwargs):
    text = _decode(data)
    # TODO: 可用 json 模块解析，按顶层列表的每条记录（或每个键值对）独立切块。
    return _split_text_like(text, source)


# XML 分片。
#
# 说明：按标签节点切更合理。当前先按文本切分。
def split_xml(data: bytes, source: str, **kwargs):
    text = _decode(data)
    # TODO: 可用 xml.etree.ElementTree 解析，按节点/子树切块。
    return _split_text_like(text, source)


# HTML 分片。
#
# 说明：先去掉标签得到正文，再切分。
# 依赖：可用标准库 html.parser 或第三方 beautifulsoup4。当前先用纯文本。
def split_html(data: bytes, source: str, **kwargs):
    text = _decode(data)
    # TODO: 剥离 HTML 标签与脚本/样式，保留正文后调用 _split_text_like。
    return _split_text_like(text, source)


# ======================================================================
# 注册表与分发
# ======================================================================

# 扩展名（小写，含点） -> 分片函数。新增格式只需在这里加一行。
_SPLITTER_REGISTRY = {
    ".txt": split_txt,
    ".md": split_md,
    ".markdown": split_md,
    ".pdf": split_pdf,
    ".docx": split_docx,
    ".xlsx": split_xlsx,
    ".csv": split_csv,
    ".json": split_json,
    ".xml": split_xml,
    ".html": split_html,
    ".htm": split_html,
}


# 根据文件名或扩展名拿到对应的分片函数。
#
# 参数：
#     filename: 文件名（含扩展名），也支持直接传 ".pdf" 这种扩展名。
#
# 返回：
#     分片函数；未知格式返回 None。
def get_splitter_for(filename: str):
    suffix = Path(filename).suffix.lower()
    if not suffix and filename.startswith("."):
        suffix = filename.lower()
    return _SPLITTER_REGISTRY.get(suffix)


# 统一入口：按文件名自动选分片函数并切分。
#
# 参数：
#     data: 原始字节。
#     filename: 文件名（用扩展名决定走哪个分片函数）。
#     **kwargs: 透传给具体分片函数的扩展参数。
#
# 返回：
#     统一格式的块列表。
#
# 异常：
#     ValueError: 不支持的格式。
def split_by_format(data: bytes, filename: str, **kwargs):
    splitter_func = get_splitter_for(filename)
    if splitter_func is None:
        raise ValueError(f"不支持的文件格式：{filename}")
    return splitter_func(data, filename, **kwargs)
