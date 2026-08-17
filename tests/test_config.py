"""测试：配置加载。"""
from pathlib import Path

from src.config import _load_yaml, load_settings


def test_load_settings_defaults():
    s = load_settings(Path("configs/config.yaml"))
    assert s.app.name == "rag-kf-support"
    assert s.compliance.max_history_rounds == 10
    assert "人工" in s.compliance.transfer_keywords


def test_config_yaml_has_no_plaintext_key():
    # 安全性质：config.yaml 不应明文写 API Key（应通过环境变量 / .env 注入）
    data = _load_yaml(Path("configs/config.yaml"))
    assert data.get("llm", {}).get("api_key", "") == ""


def test_llm_available_false_without_key(monkeypatch):
    # 无环境变量、无 .env 时，available 应为 False（用临时空 .env 隔离）
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    from src import config as config_module

    monkeypatch.setattr(config_module, "_load_dotenv", lambda *a, **k: {})
    s = load_settings(Path("configs/config.yaml"))
    assert s.llm.available is False
