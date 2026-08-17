"""配置加载：基于 pydantic-settings，支持 YAML 文件 + 环境变量覆盖。

加载优先级：环境变量 > .env 文件 > YAML 配置文件。
敏感凭证（API Key、管理员令牌）建议通过环境变量或 .env 注入，勿明文提交。

Author: MADENG
Reviewer: Li Rongdong
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


# 应用基础配置。
#
# Attributes:
#     name: 应用名。
#     env: 运行环境（dev/prod）。
#     log_level: 日志级别。
#     host: 监听地址。
#     port: 监听端口。
class AppConfig(BaseSettings):
    name: str = "rag-kf-support"
    env: str = "dev"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000


# 向量库配置。
#
# Attributes:
#     type: 向量库类型（当前固定 chroma）。
#     host: 远程向量库地址（未使用）。
#     port: 远程向量库端口（未使用）。
#     collection: 集合名。
#     persist_dir: 持久化目录。
class VectorDBConfig(BaseSettings):
    type: str = "chroma"
    host: str = ""
    port: int = 0
    collection: str = "kf_knowledge"
    persist_dir: str = "data/chroma"


# Embedding 模型配置。
#
# Attributes:
#     provider: 提供方（当前固定 local）。
#     model: 模型名。
#     dimension: 向量维度。
#     device: 计算设备（cpu/cuda）。
#
# Notes:
#     采用本地 sentence-transformers 模型（bge-small-zh-v1.5：约 95MB、512 维，
#     轻量中文模型，CPU 可跑），避免依赖外部 API。
class EmbeddingConfig(BaseSettings):
    provider: str = "local"  # local
    model: str = "BAAI/bge-small-zh-v1.5"
    dimension: int = 512
    device: str = "cpu"  # cpu / cuda


# LLM 配置（OpenAI 兼容协议）。
#
# Attributes:
#     provider: 提供方（deepseek/openai/qwen/hunyuan）。
#     api_key: API Key（环境变量 LLM_API_KEY；为空则走降级兜底）。
#     base_url: API base URL。
#     model: 模型名。
#     temperature: 采样温度。
#     timeout_seconds: 请求超时。
#     system_role: 系统角色名（调用 LLM 时作为 messages[0].role="system"）。
class LLMConfig(BaseSettings):
    provider: str = "deepseek"  # deepseek / qwen / hunyuan
    api_key: str = ""  # 环境变量 LLM_API_KEY；为空则走降级兜底
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.3
    timeout_seconds: int = 30
    # 遵循 OpenAI 兼容 messages 规范，详见 README「LLM 调用 role 规范」一节。
    system_role: str = "知识库助手"

    @property
    def available(self) -> bool:
        # 是否配置了可用 LLM（无 Key 时走降级）。
        #
        # Returns:
        #     True 表示已配置 API Key，可调用 LLM；否则走降级模式。
        return bool(self.api_key)


# 检索配置。
#
# Attributes:
#     top_k: 检索返回的最大文档数。
#     threshold: 相似度阈值，低于此值判定未命中。
class RetrievalConfig(BaseSettings):
    top_k: int = 5
    threshold: float = 0.5


# 合规配置。
#
# Attributes:
#     sensitive_words: 敏感词列表。
#     transfer_keywords: 转人工关键词。
#     max_history_rounds: 保留的最大历史轮数。
class ComplianceConfig(BaseSettings):
    sensitive_words: list[str] = []
    transfer_keywords: list[str] = ["人工", "转人工", "投诉"]
    max_history_rounds: int = 10


# 管理员鉴权配置。
#
# Attributes:
#     token: 管理员令牌（通过环境变量 ADMIN_TOKEN 或 .env 注入，勿明文写入 YAML）。
#
# Notes:
#     令牌用于保护 /kb/* 管理接口（上传/分片/入库/预览原文）。
#     为空则表示关闭鉴权（仅建议本地开发使用）。
class AdminConfig(BaseSettings):
    token: str = ""


# 全局配置聚合。
#
# Attributes:
#     app: 应用基础配置。
#     vector_db: 向量库配置。
#     embedding: Embedding 配置。
#     llm: LLM 配置。
#     retrieval: 检索配置。
#     compliance: 合规配置。
#     admin: 管理员鉴权配置。
class Settings(BaseSettings):
    app: AppConfig = AppConfig()
    vector_db: VectorDBConfig = VectorDBConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    compliance: ComplianceConfig = ComplianceConfig()
    admin: AdminConfig = AdminConfig()

    model_config = SettingsConfigDict(env_nested_delimiter="__")


# 读取 YAML 配置文件为字典。
#
# Args:
#     path: 配置文件路径。
#
# Returns:
#     解析后的字典；文件不存在或内容为空时返回空字典。
def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# 嵌套环境变量映射：env key -> (section, field)
# 用于把 LLM__TEMPERATURE 这类环境变量合并到 Pydantic Settings 的嵌套结构中。
_NESTED_ENV_KEYS: dict[str, tuple[str, str]] = {
    "LLM__PROVIDER": ("llm", "provider"),
    "LLM__API_KEY": ("llm", "api_key"),
    "LLM__BASE_URL": ("llm", "base_url"),
    "LLM__MODEL": ("llm", "model"),
    "LLM__TEMPERATURE": ("llm", "temperature"),
    "LLM__TIMEOUT_SECONDS": ("llm", "timeout_seconds"),
    "LLM__SYSTEM_ROLE": ("llm", "system_role"),
    "EMBEDDING__PROVIDER": ("embedding", "provider"),
    "EMBEDDING__MODEL": ("embedding", "model"),
    "EMBEDDING__DIMENSION": ("embedding", "dimension"),
    "EMBEDDING__DEVICE": ("embedding", "device"),
    "RETRIEVAL__TOP_K": ("retrieval", "top_k"),
    "RETRIEVAL__THRESHOLD": ("retrieval", "threshold"),
    "VECTOR_DB__TYPE": ("vector_db", "type"),
    "VECTOR_DB__HOST": ("vector_db", "host"),
    "VECTOR_DB__PORT": ("vector_db", "port"),
    "VECTOR_DB__COLLECTION": ("vector_db", "collection"),
    "VECTOR_DB__PERSIST_DIR": ("vector_db", "persist_dir"),
    "ADMIN_TOKEN": ("admin", "token"),
    # 兼容旧式单层键：LLM_API_KEY（不依赖嵌套分隔符）
    "LLM_API_KEY": ("llm", "api_key"),
}

# 数值字段：做 int/float 类型转换
_INT_FIELDS = {"timeout_seconds", "dimension", "port", "top_k"}
_FLOAT_FIELDS = {"temperature", "threshold"}


# 将环境变量字符串值转换为目标字段类型。
#
# Args:
#     value: 原始字符串值。
#     field: 字段名，用于判断目标类型。
#
# Returns:
#     按字段类型转换后的值（int / float / str）。
#
# Notes:
#     数值字段类型由 ``_INT_FIELDS`` 与 ``_FLOAT_FIELDS`` 集合决定。
def _coerce(value: str, field: str):
    if field in _INT_FIELDS:
        return int(value)
    if field in _FLOAT_FIELDS:
        return float(value)
    return value


# 加载配置：YAML 文件为基底，环境变量 + .env 覆盖敏感项。
#
# Args:
#     config_path: 可选配置文件路径，默认 ``configs/config.yaml``。
#
# Returns:
#     合并后的全局配置实例。
#
# Notes:
#     每次调用都重读（无缓存），便于 PATCH /kb/config 后下个请求拿到最新值。
def load_settings(config_path: str | Path | None = None) -> Settings:
    path = Path(config_path) if config_path else Path("configs/config.yaml")
    data = _load_yaml(path)

    # 合并 .env 文件（简单解析，不引依赖）
    dotenv = _load_dotenv()

    # 合并 env var + dotenv：env var 优先
    merged: dict[str, str] = {**dotenv, **{k: v for k, v in os.environ.items()}}

    # 把白名单嵌套键合并到 Pydantic Settings 结构
    for env_key, (section, field) in _NESTED_ENV_KEYS.items():
        val = merged.get(env_key)
        if val is None:
            continue
        data.setdefault(section, {})[field] = _coerce(val, field)

    return Settings(**data)


# 读取 .env 文件（简单解析，不引入额外依赖）。
#
# Args:
#     path: 可选 .env 文件路径，默认项目根目录 ``.env``。
#
# Returns:
#     键值字典；文件不存在时返回空字典。
#
# Notes:
#     优先级：环境变量 > .env 文件 > YAML 配置。
def _load_dotenv(path: str | Path | None = None) -> dict[str, str]:
    dotenv_path = Path(path) if path else Path(".env")
    if not dotenv_path.exists():
        return {}
    result: dict[str, str] = {}
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            result[key] = value
    return result
