"""配置加载：基于 pydantic-settings，支持 YAML 文件 + 环境变量覆盖。"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    name: str = "rag-kf-support"
    env: str = "dev"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000


class WecomCallbackConfig(BaseSettings):
    token: str = ""
    encoding_aes_key: str = ""


class WecomConfig(BaseSettings):
    corpid: str = ""
    secret: str = ""
    agentid: int = 0
    callback: WecomCallbackConfig = WecomCallbackConfig()

    model_config = SettingsConfigDict(env_prefix="WECOM_", env_nested_delimiter="__")


class RedisConfig(BaseSettings):
    url: str = "redis://localhost:6379/0"


class VectorDBConfig(BaseSettings):
    type: str = "chroma"
    host: str = ""
    port: int = 0
    collection: str = "kf_knowledge"


class EmbeddingConfig(BaseSettings):
    model: str = "BAAI/bge-m3"
    dimension: int = 1024


class LLMConfig(BaseSettings):
    provider: str = "deepseek"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.3
    timeout_seconds: int = 30


class ComplianceConfig(BaseSettings):
    sensitive_words: list[str] = []
    transfer_keywords: list[str] = ["人工", "转人工", "投诉"]
    max_answers_per_48h: int = 5


class Settings(BaseSettings):
    app: AppConfig = AppConfig()
    wecom: WecomConfig = WecomConfig()
    redis: RedisConfig = RedisConfig()
    vector_db: VectorDBConfig = VectorDBConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()
    compliance: ComplianceConfig = ComplianceConfig()

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
    return Settings(**data)
