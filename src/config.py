"""配置加载：基于 pydantic-settings，支持 YAML 文件 + 环境变量覆盖。"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    name: str = "rag-kf-support"
    env: str = "dev"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000


class VectorDBConfig(BaseSettings):
    type: str = "chroma"
    host: str = ""
    port: int = 0
    collection: str = "kf_knowledge"
    persist_dir: str = "data/chroma"


class EmbeddingConfig(BaseSettings):
    # 本地 sentence-transformers 模型，避免依赖外部 API
    # bge-small-zh-v1.5：约 95MB、512 维，轻量中文模型，CPU 可跑
    provider: str = "local"  # local
    model: str = "BAAI/bge-small-zh-v1.5"
    dimension: int = 512
    device: str = "cpu"  # cpu / cuda


class LLMConfig(BaseSettings):
    provider: str = "deepseek"  # deepseek / qwen / hunyuan
    api_key: str = ""  # 环境变量 LLM_API_KEY；为空则走降级兜底
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.3
    timeout_seconds: int = 30
    # 系统角色：调用 LLM 时 messages[0].role="system"，content 以该角色身份进行回复。
    # 遵循 OpenAI 兼容 messages 规范，详见 README「LLM 调用 role 规范」一节。
    system_role: str = "知识库助手"

    @property
    def available(self) -> bool:
        """是否配置了可用 LLM（无 Key 时走降级）。"""
        return bool(self.api_key)


class RetrievalConfig(BaseSettings):
    top_k: int = 5
    threshold: float = 0.5  # 相似度低于阈值则判定未命中


class ComplianceConfig(BaseSettings):
    sensitive_words: list[str] = []
    transfer_keywords: list[str] = ["人工", "转人工", "投诉"]
    max_history_rounds: int = 10


class AdminConfig(BaseSettings):
    # 管理员令牌：用于保护 /kb/* 管理接口（上传/分片/入库/预览原文）。
    # 通过环境变量 ADMIN_TOKEN 或 .env 注入，勿明文写入 YAML。
    # 为空则表示关闭鉴权（仅建议本地开发使用）。
    token: str = ""


class Settings(BaseSettings):
    app: AppConfig = AppConfig()
    vector_db: VectorDBConfig = VectorDBConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    compliance: ComplianceConfig = ComplianceConfig()
    admin: AdminConfig = AdminConfig()

    model_config = SettingsConfigDict(env_nested_delimiter="__")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(config_path: str | Path | None = None) -> Settings:
    """加载配置：YAML 文件为基底，环境变量覆盖敏感项。"""
    path = Path(config_path) if config_path else Path("configs/config.yaml")
    data = _load_yaml(path)

    # 敏感项优先从环境变量注入（勿明文写进 YAML）
    api_key = os.getenv("LLM_API_KEY") or _load_dotenv().get("LLM_API_KEY")
    if api_key:
        data.setdefault("llm", {})["api_key"] = api_key

    admin_token = os.getenv("ADMIN_TOKEN") or _load_dotenv().get("ADMIN_TOKEN")
    if admin_token:
        data.setdefault("admin", {})["token"] = admin_token

    return Settings(**data)


def _load_dotenv(path: str | Path | None = None) -> dict[str, str]:
    """读取 .env 文件（简单解析，不引入额外依赖）。

    优先级：环境变量 > .env 文件 > YAML 配置。
    """
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
