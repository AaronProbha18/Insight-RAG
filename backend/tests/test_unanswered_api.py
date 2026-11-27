import psycopg
import pytest
from fastapi.testclient import TestClient

from plumbline.api.app import app, get_conn
from plumbline.feedback import FeedbackRequest, record_answer, save_feedback

pytestmark = pytest.mark.db


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


def _add(db, question, *, abstained, rating=None, age_days=0):
    pid = record_answer(
        db,
        question,
        {"answer": "SECRET-ANSWER", "citations": [], "abstained": abstained},
        chat_model="m",
        embedding_model="e",
        threshold=0.3,
    )
    db.execute(
        "UPDATE answers SET created_at = now() - make_interval(days => %s)"
        " WHERE public_id = %s",
        (age_days, pid),
    )
    if rating:
        save_feedback(
            db, FeedbackRequest(answer_id=pid, rating=rating, reason="wrong_source")
        )


def test_shape_and_no_internals(client, db):
    _add(db, "Refused?", abstained=True)
    _add(db, "Wrong?", abstained=False, rating="down")

    r = client.get("/unanswered")

    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 2
    for item in items:
        assert set(item) == {
            "question",
            "count",
            "last_asked_at",
            "reason",
            "feedback_reason",
        }
    for leak in ("score", "confidence", "ans_", "public_id", "SECRET-ANSWER"):
        assert leak not in r.text


def test_days_filters_the_range(client, db):
    _add(db, "Old?", abstained=True, age_days=40)

    assert client.get("/unanswered?days=30").json() == {"items": []}
    assert len(client.get("/unanswered?days=90").json()["items"]) == 1
    assert len(client.get("/unanswered").json()["items"]) == 1


@pytest.mark.parametrize("days", ["5", "abc"])
def test_invalid_days_is_422(client, days):
    assert client.get(f"/unanswered?days={days}").status_code == 422
