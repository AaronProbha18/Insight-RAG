import re
import time
from pathlib import Path

import psycopg
import pytest
from fake_embeddings import FailingEmbeddings, FakeEmbeddings

from plumbline.connectors import ParseError
from plumbline.connectors.builtin.docx import DocxConnector
from plumbline.connectors.builtin.pdf import PdfConnector
from plumbline.ingestion import UnsupportedFileType, ingest, new_document_id

EVALS = Path(__file__).parents[2] / "evals" / "connectors"
PDF = (EVALS / "pdf" / "corpus" / "returns-policy.pdf").read_bytes()
DOCX = (EVALS / "docx" / "corpus" / "employee-handbook.docx").read_bytes()
CONNECTORS = {"pdf": PdfConnector(), "docx": DocxConnector()}
DOC_ID = re.compile(r"^doc_[0-9A-HJKMNP-TV-Z]{26}$")


@pytest.fixture
def conn(migrate, database_url):
    migrate()
    with psycopg.connect(database_url, autocommit=True) as c:
        yield c


def _counts(conn):
    return conn.execute(
        "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)"
    ).fetchone()


@pytest.mark.db
def test_pdf_is_indexed(conn):
    result = ingest(conn, PDF, "returns.pdf", CONNECTORS, FakeEmbeddings())

    assert result.created and result.connector == "pdf" and result.chunks > 0
    assert DOC_ID.match(result.document_id)
    locators = [r[0] for r in conn.execute("SELECT locator FROM chunks")]
    assert locators and all(re.fullmatch(r"p\. \d+", loc) for loc in locators)


@pytest.mark.db
def test_docx_is_indexed(conn):
    result = ingest(conn, DOCX, "handbook.docx", CONNECTORS, FakeEmbeddings())

    assert result.created and result.connector == "docx" and result.chunks > 0


@pytest.mark.db
def test_same_bytes_twice_is_idempotent(conn):
    first = ingest(conn, PDF, "a.pdf", CONNECTORS, FakeEmbeddings())
    embeddings = FakeEmbeddings()

    second = ingest(conn, PDF, "renamed.pdf", CONNECTORS, embeddings)

    assert second.document_id == first.document_id and not second.created
    assert second.chunks == first.chunks
    assert _counts(conn) == (1, first.chunks)
    assert embeddings.calls == 0


@pytest.mark.db
def test_embedding_failure_stores_nothing(conn):
    with pytest.raises(ConnectionError):
        ingest(conn, PDF, "a.pdf", CONNECTORS, FailingEmbeddings())

    assert _counts(conn) == (0, 0)


@pytest.mark.db
def test_insert_failure_rolls_back(conn):
    with pytest.raises(psycopg.errors.DataException, match="dimensions"):
        ingest(conn, PDF, "a.pdf", CONNECTORS, FakeEmbeddings(dim=4))

    assert _counts(conn) == (0, 0)


@pytest.mark.db
def test_unsupported_type(conn):
    with pytest.raises(UnsupportedFileType, match=r"\.docx, \.pdf"):
        ingest(conn, b"\x89PNG", "photo.png", CONNECTORS, FakeEmbeddings())

    assert _counts(conn) == (0, 0)


@pytest.mark.db
def test_corrupt_pdf_stores_nothing(conn):
    with pytest.raises(ParseError):
        ingest(conn, b"not a pdf", "bad.pdf", CONNECTORS, FakeEmbeddings())

    assert _counts(conn) == (0, 0)


def test_new_document_id_format_and_order():
    a = new_document_id()
    time.sleep(0.002)
    b = new_document_id()

    assert DOC_ID.match(a) and DOC_ID.match(b)
    assert a < b
