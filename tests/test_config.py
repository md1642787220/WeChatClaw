"""测试：配置加载。"""
from pathlib import Path

from src.config import load_settings


def test_load_settings_defaults():
    s = load_settings(Path("configs/config.yaml"))
    assert s.app.name == "rag-kf-support"
    assert s.compliance.max_answers_per_48h == 5
    assert "人工" in s.compliance.transfer_keywords
