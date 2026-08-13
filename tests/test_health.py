"""测试：健康检查。"""
import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
