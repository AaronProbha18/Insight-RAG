import importlib

from fastapi.testclient import TestClient

import plumbline.api.app as app_module


def _client(monkeypatch, ui_dir=None):
    if ui_dir:
        monkeypatch.setenv("INSIGHT_RAG_UI_DIR", str(ui_dir))
    else:
        monkeypatch.delenv("INSIGHT_RAG_UI_DIR", raising=False)
    return TestClient(importlib.reload(app_module).app)


def test_serves_ui_when_dir_set(monkeypatch, tmp_path):
    (tmp_path / "index.html").write_text("<title>insight-rag</title>")

    client = _client(monkeypatch, tmp_path)

    assert "<title>insight-rag</title>" in client.get("/").text
    assert client.get("/health").json() == {"status": "ok"}


def test_no_ui_mount_by_default(monkeypatch):
    client = _client(monkeypatch)

    assert client.get("/").status_code == 404
    assert client.get("/health").json() == {"status": "ok"}
