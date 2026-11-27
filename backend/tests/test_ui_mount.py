import importlib

from fastapi.testclient import TestClient

import plumbline.api.app as app_module


def _client(monkeypatch, ui_dir=None):
    if ui_dir:
        monkeypatch.setenv("PLUMBLINE_UI_DIR", str(ui_dir))
    else:
        monkeypatch.delenv("PLUMBLINE_UI_DIR", raising=False)
    return TestClient(importlib.reload(app_module).app)


def test_serves_ui_when_dir_set(monkeypatch, tmp_path):
    (tmp_path / "index.html").write_text("<title>Plumbline</title>")

    client = _client(monkeypatch, tmp_path)

    assert "<title>Plumbline</title>" in client.get("/").text
    assert client.get("/health").json() == {"status": "ok"}


class _NoRows:
    def execute(self, *_args):
        return self

    def fetchall(self):
        return []


def test_app_path_serves_index(monkeypatch, tmp_path):
    (tmp_path / "index.html").write_text("<title>Plumbline</title>")

    client = _client(monkeypatch, tmp_path)

    for path in ("/app/documents", "/app/connectors", "/app/chats/conv_X"):
        response = client.get(path)

        assert response.status_code == 200
        assert "<title>Plumbline</title>" in response.text


def test_api_and_unknown_paths_are_not_index(monkeypatch, tmp_path):
    (tmp_path / "index.html").write_text("<title>Plumbline</title>")
    client = _client(monkeypatch, tmp_path)
    client.app.dependency_overrides[app_module.get_conn] = lambda: _NoRows()

    assert client.get("/nope").status_code == 404
    documents = client.get("/documents")
    assert documents.json() == []
    assert "<title>" not in documents.text
    connectors = client.get("/connectors")
    assert {c["name"] for c in connectors.json()} >= {"pdf", "docx"}
    assert "<title>" not in connectors.text


def test_no_app_route_without_ui_dir(monkeypatch):
    assert _client(monkeypatch).get("/app/documents").status_code == 404


def test_no_ui_mount_by_default(monkeypatch):
    client = _client(monkeypatch)

    assert client.get("/").status_code == 404
    assert client.get("/health").json() == {"status": "ok"}
