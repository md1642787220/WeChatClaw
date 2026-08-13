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


class VectorDBConfig(BaseSettings):
    type: str = "chroma"
    host: str = ""
    port: int = 0
    collection: str = "kf_knowledge"
    persist_dir: str = "data/chroma"


class EmbeddingConfig(BaseSettings):
    # 首期使用本地 sentence-transformers 模型，避免依赖外部 API
    provider: str = "local"  # local / openai
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


class Settings(BaseSettings):
    app: AppConfig = AppConfig()
    vector_db: VectorDBConfig = VectorDBConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
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
