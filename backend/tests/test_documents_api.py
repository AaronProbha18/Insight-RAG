import re
from pathlib import Path

import psycopg
import pytest
from fake_embeddings import FailingEmbeddings, FakeEmbeddings
from fastapi.testclient import TestClient

from plumbline.api.app import app, get_conn, get_embeddings

pytestmark = pytest.mark.db

EVALS = Path(__file__).parents[2] / "evals" / "connectors"
PDF = (EVALS / "pdf" / "corpus" / "returns-policy.pdf").read_bytes()
DOCX = (EVALS / "docx" / "corpus" / "employee-handbook.docx").read_bytes()
DOC_ID = re.compile(r"^doc_[0-9A-HJKMNP-TV-Z]{26}$")


@pytest.fixture
def client(migrate, database_url):
    migrate()

    def conn():
        with psycopg.connect(database_url, autocommit=True) as c:
            yield c

    app.dependency_overrides[get_conn] = conn
    app.dependency_overrides[get_embeddings] = FakeEmbeddings
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def _upload(client, name, data):
    return client.post("/documents", files={"file": (name, data)})


def test_pdf_upload_201(client):
    r = _upload(client, "returns.pdf", PDF)

    assert r.status_code == 201
    body = r.json()
    assert body["connector"] == "pdf" and body["chunks"] > 0 and body["created"] is True
    assert DOC_ID.match(body["document_id"])


def test_docx_upload_201(client):
    r = _upload(client, "handbook.docx", DOCX)

    assert r.status_code == 201
    assert r.json()["connector"] == "docx" and r.json()["chunks"] > 0


def test_duplicate_upload_200(client):
    first = _upload(client, "a.pdf", PDF).json()

    r = _upload(client, "renamed.pdf", PDF)

    assert r.status_code == 200
    assert r.json()["document_id"] == first["document_id"]
    assert r.json()["created"] is False


def test_embedding_failure_500_stores_nothing(client):
    app.dependency_overrides[get_embeddings] = FailingEmbeddings

    r = _upload(client, "a.pdf", PDF)

    assert r.status_code == 500
    assert client.get("/documents").json() == []


def test_png_415_names_types(client):
    r = _upload(client, "photo.png", b"\x89PNG\r\n")

    assert r.status_code == 415
    assert ".docx, .pdf" in r.json()["detail"]


def test_rejections_work_without_a_model_configured(client, monkeypatch):
    del app.dependency_overrides[get_embeddings]
    monkeypatch.delenv("PLUMBLINE_EMBEDDINGS", raising=False)

    assert _upload(client, "photo.png", b"\x89PNG").status_code == 415
    assert _upload(client, "bad.pdf", b"not a pdf").status_code == 422


def test_corrupt_pdf_422_stores_nothing(client):
    r = _upload(client, "bad.pdf", b"not a pdf")

    assert r.status_code == 422
    assert client.get("/documents").json() == []


def test_list_documents(client):
    _upload(client, "a.pdf", PDF)
    _upload(client, "b.docx", DOCX)

    docs = client.get("/documents").json()

    assert len(docs) == 2
    assert all(DOC_ID.match(d["document_id"]) for d in docs)
    assert all(
        set(d) == {"document_id", "title", "connector", "created_at"} for d in docs
    )
