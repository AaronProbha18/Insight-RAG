import json

import psycopg
import pytest
from fastapi.testclient import TestClient

from plumbline.api.app import app, get_conn
from plumbline.feedback import record_answer

pytestmark = pytest.mark.db

ANSWERED = {
    "answer": "14 days [c1]",
    "abstained": False,
    "citations": [
        {
            "document_id": "doc_1",
            "title": "Doc 1",
            "filename": "returns.pdf",
            "locator": "p. 1",
        }
    ],
}
ABSTAINED = {"answer": "I can't answer that.", "abstained": True, "citations": []}


@pytest.fixture
def db(migrate, database_url):
    migrate()
    with psycopg.connect(database_url, autocommit=True) as c:
        yield c


@pytest.fixture
def client(db, database_url):
    def conn():
        with psycopg.connect(database_url, autocommit=True) as c:
            yield c

    app.dependency_overrides[get_conn] = conn
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def _answer(db, result=ANSWERED):
    return record_answer(
        db, "Refund window?", result, chat_model="m", embedding_model="e", threshold=0.3
    )


def _rows(db):
    return db.execute(
        "SELECT a.public_id, f.rating, f.reason, f.comment"
        " FROM feedback f JOIN answers a ON a.id = f.answer_id"
    ).fetchall()


def _send(client, answer_id, rating, reason=None, comment=None):
    return client.post(
        "/feedback",
        json={
            "answer_id": answer_id,
            "rating": rating,
            "reason": reason,
            "comment": comment,
        },
    )


def test_thumbs_up(client, db):
    aid = _answer(db)

    r = _send(client, aid, "up")

    assert r.status_code == 204 and r.content == b""
    assert _rows(db) == [(aid, "up", None, None)]


def test_thumbs_down_keeps_reason_comment_and_the_answer_record(client, db):
    aid = _answer(db)

    assert _send(client, aid, "down", "wrong_source", "wrong doc").status_code == 204

    assert _rows(db) == [(aid, "down", "wrong_source", "wrong doc")]
    question, answer, citations = db.execute(
        "SELECT question, answer, citations FROM answers WHERE public_id = %s", (aid,)
    ).fetchone()
    assert (question, answer) == ("Refund window?", "14 days [c1]")
    assert citations[0]["document_id"] == "doc_1" and citations[0]["locator"] == "p. 1"


def test_feedback_on_an_abstention(client, db):
    aid = _answer(db, ABSTAINED)

    assert _send(client, aid, "down", "should_have_answered").status_code == 204

    (abstained,) = db.execute(
        "SELECT a.abstained FROM feedback f JOIN answers a ON a.id = f.answer_id"
    ).fetchone()
    assert abstained is True
    assert _rows(db) == [(aid, "down", "should_have_answered", None)]


def test_changing_feedback_replaces_it(client, db):
    aid = _answer(db)

    _send(client, aid, "up")
    _send(client, aid, "down", "wrong", "no")
    assert _rows(db) == [(aid, "down", "wrong", "no")]
    _send(client, aid, "up")

    assert _rows(db) == [(aid, "up", None, None)]


@pytest.mark.parametrize(
    "rating, reason, comment",
    [
        ("sideways", None, None),
        ("down", "bogus", None),
        ("up", "wrong", None),
        ("down", None, "x" * 2001),
    ],
)
def test_invalid_feedback_422_stores_nothing(client, db, rating, reason, comment):
    aid = _answer(db)

    r = _send(client, aid, rating, reason, comment)

    assert r.status_code == 422
    assert _rows(db) == []
    for leaked in ("Traceback", "psycopg", "score", "confidence"):
        assert leaked not in r.text


def test_comment_of_2000_chars_is_accepted(client, db):
    assert _send(client, _answer(db), "down", None, "x" * 2000).status_code == 204


def test_unknown_answer_404_reveals_nothing(client, db):
    r = _send(client, "ans_nope", "up")

    assert r.status_code == 404
    assert r.json() == {"detail": "Answer not found"}
    assert _rows(db) == []
    assert "score" not in json.dumps(r.json())
