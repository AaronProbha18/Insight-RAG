import hashlib
from pathlib import Path

import psycopg
import pytest
import yaml

from plumbline import feedback as fb
from plumbline.feedback import FeedbackRequest, record_answer, save_feedback

pytestmark = pytest.mark.db

GOLDEN = sorted(
    (Path(__file__).parents[2] / "evals" / "connectors").glob("*/golden.yaml")
)
CITE = {
    "document_id": "doc_1",
    "title": "Handbook",
    "filename": "employee-handbook.docx",
    "locator": "p. 2",
}
ANSWERED = {"answer": "25 days [c1]", "abstained": False, "citations": [CITE]}
ABSTAINED = {"answer": "I can't answer that.", "abstained": True, "citations": []}


@pytest.fixture
def conn(migrate, database_url):
    migrate()
    with psycopg.connect(database_url, autocommit=True) as c:
        yield c


def _give(conn, result, rating, reason=None, comment=None):
    aid = record_answer(
        conn,
        "How much leave?",
        result,
        chat_model="m",
        embedding_model="e",
        threshold=0.3,
    )
    req = FeedbackRequest(answer_id=aid, rating=rating, reason=reason, comment=comment)
    assert save_feedback(conn, req)
    return aid


@pytest.mark.parametrize(
    "result, rating, reason, answerable, stems",
    [
        (ANSWERED, "up", None, True, [("employee-handbook", "p. 2")]),
        (ABSTAINED, "up", None, False, []),
        (ABSTAINED, "down", "should_have_answered", True, []),
        (ANSWERED, "down", "wrong_source", None, [("employee-handbook", "p. 2")]),
        (ANSWERED, "down", None, None, [("employee-handbook", "p. 2")]),
    ],
)
def test_mapping_rows(conn, result, rating, reason, answerable, stems):
    aid = _give(conn, result, rating, reason, "c")

    (item,) = fb.export_candidates(conn)

    assert item["id"] == f"fb-{aid}"
    assert item["answerable"] is answerable
    assert [(s["document"], s["locator"]) for s in item["supporting"]] == stems
    assert item["reference_answer"] == "" and item["key_fact"] == ""
    assert item["feedback"] == {
        "rating": rating,
        "reason": reason,
        "comment": "c",
        "answer": result["answer"],
        "chat_model": "m",
        "embedding_model": "e",
        "abstain_threshold": 0.3,
    }


def test_main_writes_golden_shaped_yaml_and_leaves_golden_sets_alone(conn, tmp_path):
    _give(conn, ANSWERED, "up")
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in GOLDEN]
    out = tmp_path / "sub" / "candidates.yaml"

    assert fb.main(["--out", str(out)]) == 0

    (item,) = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert {
        "id",
        "question",
        "answerable",
        "reference_answer",
        "supporting",
    } <= item.keys()
    assert item["supporting"] == [{"document": "employee-handbook", "locator": "p. 2"}]
    assert GOLDEN and before == [
        hashlib.sha256(p.read_bytes()).hexdigest() for p in GOLDEN
    ]


def test_export_works_after_the_cited_document_is_deleted(conn):
    (doc_id,) = conn.execute(
        "INSERT INTO documents (public_id, content_hash, title, filename, connector)"
        " VALUES ('doc_1', repeat('a', 64), 'Handbook', 'employee-handbook.docx', 'docx')"
        " RETURNING id"
    ).fetchone()
    _give(conn, ANSWERED, "up")
    conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))

    (item,) = fb.export_candidates(conn)

    assert item["supporting"] == [{"document": "employee-handbook", "locator": "p. 2"}]


def test_no_feedback_exports_an_empty_list(conn, tmp_path):
    out = tmp_path / "c.yaml"

    assert fb.main(["--out", str(out)]) == 0

    assert yaml.safe_load(out.read_text(encoding="utf-8")) == []
