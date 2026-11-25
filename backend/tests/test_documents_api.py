import json
import re
from pathlib import Path

import psycopg
import pytest
from fake_connectors import GoodConnector
from fake_embeddings import FailingEmbeddings, FakeEmbeddings
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from plumbline.api.app import app, get_conn, get_connectors, get_embeddings, get_llm
from plumbline.retrieval import retrieve

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


def _chunk_count(database_url):
    with psycopg.connect(database_url) as c:
        return c.execute("SELECT count(*) FROM chunks").fetchone()[0]


def test_delete_document_204_removes_document_and_chunks(client, database_url):
    gone = _upload(client, "a.pdf", PDF).json()
    kept = _upload(client, "b.docx", DOCX).json()

    r = client.delete(f"/documents/{gone['document_id']}")

    assert r.status_code == 204 and r.content == b""
    assert [d["document_id"] for d in client.get("/documents").json()] == [
        kept["document_id"]
    ]
    assert _chunk_count(database_url) == kept["chunks"]


def test_delete_unknown_and_repeat_404(client):
    missing = client.delete("/documents/doc_" + "0" * 26)
    doc = _upload(client, "a.pdf", PDF).json()
    assert client.delete(f"/documents/{doc['document_id']}").status_code == 204

    again = client.delete(f"/documents/{doc['document_id']}")

    assert missing.status_code == again.status_code == 404
    assert missing.json() == again.json() == {"detail": "Document not found"}


def test_reupload_after_delete_201(client):
    doc = _upload(client, "a.pdf", PDF).json()
    client.delete(f"/documents/{doc['document_id']}")

    r = _upload(client, "a.pdf", PDF)

    assert r.status_code == 201 and r.json()["created"] is True


class KeywordEmbeddings:
    """'zebra' texts and queries share one axis; everything else is orthogonal to it."""

    def _vector(self, text):
        return (
            [1.0, 0.0] + [0.0] * 6
            if "zebra" in text.lower()
            else [0.0, 1.0] + [0.0] * 6
        )

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


class ScriptedLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return AIMessage("Zebras sleep standing up [c1].")


def _ask(client, question):
    r = client.post("/chat", json={"messages": [{"role": "user", "content": question}]})
    return [
        (name.removeprefix("event: "), json.loads(data.removeprefix("data: ")))
        for name, data in (b.split("\n") for b in r.text.strip().split("\n\n"))
    ]


def test_deleted_document_is_never_retrieved_or_cited(client, database_url):
    llm = ScriptedLLM()
    app.dependency_overrides[get_embeddings] = KeywordEmbeddings
    app.dependency_overrides[get_connectors] = lambda: {"good": GoodConnector()}
    app.dependency_overrides[get_llm] = lambda: llm
    zebra = _upload(client, "zebras.txt", b"Zebras sleep standing up.").json()
    _upload(client, "other.txt", b"Invoices are due in 30 days.")

    before = _ask(client, "How do zebras sleep?")
    assert [
        c["document_id"] for n, d in before if n == "citations" for c in d["citations"]
    ] == [zebra["document_id"]]
    assert llm.calls == 1

    assert client.delete(f"/documents/{zebra['document_id']}").status_code == 204

    after = _ask(client, "How do zebras sleep?")
    assert [n for n, _ in after] == ["abstain", "done"]
    assert llm.calls == 1  # best remaining score is below the threshold
    with psycopg.connect(database_url) as conn:
        hits = retrieve(conn, KeywordEmbeddings(), "zebra", 5)
    assert zebra["document_id"] not in {h.document_id for h in hits}
