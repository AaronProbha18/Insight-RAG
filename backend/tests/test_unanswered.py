import psycopg
import pytest

from plumbline.feedback import FeedbackRequest, record_answer, save_feedback
from plumbline.unanswered import list_groups

pytestmark = pytest.mark.db


@pytest.fixture
def conn(migrate, database_url):
    migrate()
    with psycopg.connect(database_url, autocommit=True) as c:
        yield c


def _add(conn, question, *, abstained=True, rating=None, reason=None, age_days=0):
    pid = record_answer(
        conn,
        question,
        {"answer": "x", "citations": [], "abstained": abstained},
        chat_model="m",
        embedding_model="e",
        threshold=0.3,
    )
    conn.execute(
        "UPDATE answers SET created_at = now() - make_interval(days => %s)"
        " WHERE public_id = %s",
        (age_days, pid),
    )
    if rating:
        save_feedback(
            conn, FeedbackRequest(answer_id=pid, rating=rating, reason=reason)
        )
    return pid


def test_empty(conn):
    assert list_groups(conn, 30) == []


def test_case_and_spacing_group_into_one(conn):
    _add(conn, "What is the refund window?", age_days=3)
    _add(conn, "  what is  the REFUND window? ", age_days=2)
    _add(conn, "What is the refund window?", age_days=1)
    (g,) = list_groups(conn, 30)
    assert g["count"] == 3
    assert g["question"] == "What is the refund window?"
    assert g["reason"] == "not_in_documents"
    assert g["feedback_reason"] is None
    assert set(g) == {"question", "count", "last_asked_at", "reason", "feedback_reason"}


def test_marked_wrong_carries_the_reason(conn):
    _add(conn, "Q?", abstained=False, rating="down", reason="wrong_source")
    (g,) = list_groups(conn, None)
    assert (g["reason"], g["feedback_reason"]) == ("marked_wrong", "wrong_source")


def test_unrated_and_thumbs_up_answers_are_not_listed(conn):
    _add(conn, "A?", abstained=False)
    _add(conn, "B?", abstained=False, rating="up")
    assert list_groups(conn, None) == []


def test_latest_event_reason_wins(conn):
    _add(conn, "Q?", age_days=5)
    _add(conn, "Q?", abstained=False, rating="down", reason="wrong", age_days=1)
    _add(conn, "R?", abstained=False, rating="down", reason="wrong", age_days=5)
    _add(conn, "R?", age_days=1)
    by_q = {g["question"]: g for g in list_groups(conn, 30)}
    assert by_q["Q?"]["reason"] == "marked_wrong"
    assert by_q["Q?"]["count"] == 2
    assert by_q["R?"]["reason"] == "not_in_documents"


def test_range_filters_events_before_counting(conn):
    _add(conn, "Old?", age_days=40)
    _add(conn, "Mixed?", age_days=40)
    _add(conn, "Mixed?", age_days=1)
    assert [(g["question"], g["count"]) for g in list_groups(conn, 30)] == [
        ("Mixed?", 1)
    ]
    assert len(list_groups(conn, 90)) == 2
    assert len(list_groups(conn, None)) == 2
