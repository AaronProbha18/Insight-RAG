import psycopg
import pytest

from plumbline.feedback import record_answer

pytestmark = pytest.mark.db


@pytest.fixture
def conn(migrate, database_url):
    migrate()
    with psycopg.connect(database_url, autocommit=True) as c:
        yield c


def _record(conn, result, question="Refund window?"):
    return record_answer(
        conn,
        question,
        result,
        chat_model="ollama:qwen",
        embedding_model="ollama:nomic",
        threshold=0.3,
    )


def test_answer_row_holds_everything_needed_to_review_it(conn):
    citation = {
        "id": "c1",
        "document_id": "doc_1",
        "title": "Doc 1",
        "filename": "handbook.pdf",
        "locator": "p. 1",
        "text": "not stored",
    }
    answer_id = _record(
        conn, {"answer": "14 days [c1]", "citations": [citation], "abstained": False}
    )

    row = conn.execute(
        "SELECT question, answer, abstained, citations, chat_model,"
        " embedding_model, abstain_threshold FROM answers WHERE public_id = %s",
        (answer_id,),
    ).fetchone()
    assert answer_id.startswith("ans_")
    assert row == (
        "Refund window?",
        "14 days [c1]",
        False,
        [
            {
                "document_id": "doc_1",
                "title": "Doc 1",
                "filename": "handbook.pdf",
                "locator": "p. 1",
            }
        ],
        "ollama:qwen",
        "ollama:nomic",
        0.3,
    )


def test_abstention_is_recorded_and_ids_are_distinct(conn):
    abstain = {"answer": "I can't answer that.", "citations": [], "abstained": True}
    first, second = _record(conn, abstain), _record(conn, abstain)

    rows = conn.execute("SELECT abstained, citations FROM answers").fetchall()
    assert first != second
    assert rows == [(True, []), (True, [])]
