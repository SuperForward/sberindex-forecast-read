"""Веб-сервер: REST API отвечает, запись закрыта по умолчанию."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import store  # noqa: E402

HAS_DATA = not store.load("spending_mo").empty


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("SBERINDEX_ALLOW_WRITE", raising=False)
    from fastapi.testclient import TestClient
    import server.app as srv
    monkeypatch.setattr(srv, "WRITE", False)
    return TestClient(srv.app)


def test_write_is_closed_by_default(client):
    r = client.post("/api/recalc/auto")
    assert r.status_code == 403 and "только для чтения" in r.json()["error"]
    assert client.post("/api/auto-update").json() == {"action": "off"}


def test_ui_and_docs_are_served(client):
    assert client.get("/").status_code == 200
    assert "SberIndex" in client.get("/ui/terminal.html").text
    landing = client.get("/ui/landing.html").text
    for name in ("summary", "models", "horizons", "cpd", "results", "news"):     # всё, что читает лендинг, есть в API
        assert f"/api/{name}'" in landing
    assert client.get("/docs").status_code == 200
    assert "/api/summary" in client.get("/openapi.json").json()["paths"]


@pytest.mark.skipif(not HAS_DATA, reason="нужны данные")
def test_read_endpoints(client):
    for url in ["/api/summary", "/api/models", "/api/horizons", "/api/cpd", "/api/results", "/api/status",
                "/api/mo?query=Орск", "/api/mo/53723000", "/api/mo/53723000/forecast", "/api/cpd/53723000"]:
        r = client.get(url)
        assert r.status_code == 200 and "error" not in r.json(), (url, r.text[:200])
    assert client.get("/api/status").json()["read_only"] is True
    assert client.get("/api/mo", params={"query": "Орск"}).json()["items"][0]["oktmo"] == "53723000"
