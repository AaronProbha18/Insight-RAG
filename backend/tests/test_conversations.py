import psycopg
import pytest

from plumbline import conversations
from plumbline.api.app import Recorder
from plumbline.feedback import record_answer

pytestmark = pytest.mark.db


@pytest.fixture
def conn(migrate, database_url):
    migrate()
    with psycopg.connect(database_url, autocommit=True) as c:
        yield c


def _turn(conn, conv_id, question="q", doc="doc_1"):
    citations = [
        {"document_id": doc, "title": "Doc", "filename": "f.pdf", "locator": "p. 1"}
    ]
    public_id = record_answer(
        conn,
        question,
        {"answer": "a", "citations": citations, "abstained": False},
        chat_model="m",
        embedding_model="e",
        threshold=0.3,
    )
    conn.execute(
        "UPDATE answers SET conversation_id = %s WHERE public_id = %s",
        (conv_id, public_id),
    )
    return public_id


def _document(conn, public_id="doc_1"):
    conn.execute(
        "INSERT INTO documents (public_id, content_hash, title, filename, connector)"
        " VALUES (%s, repeat('a', 64), 'Doc', 'f.pdf', 'pdf')",
        (public_id,),
    )


def test_create_titles_from_the_question(conn):
    _, public_id = conversations.create(conn, "  What  is\nthe refund window? ")

    assert public_id.startswith("conv_")
    assert conversations.get(conn, public_id)["title"] == "What is the refund window?"


@pytest.mark.parametrize(
    "question, title",
    [("x" * 200, "x" * 80 + "…"), ("x" * 80, "x" * 80)],
)
def test_create_cuts_long_titles(conn, question, title):
    _, public_id = conversations.create(conn, question)

    assert conversations.get(conn, public_id)["title"] == title


def test_resolve(conn):
    conv_id, public_id = conversations.create(conn, "q")

    assert conversations.resolve(conn, public_id) == conv_id
    assert conversations.resolve(conn, "conv_missing") is None


def test_touch_moves_a_conversation_to_the_top(conn):
    first, first_id = conversations.create(conn, "first")
    conversations.create(conn, "second")
    assert conversations.list_all(conn)[0]["title"] == "second"

    conversations.touch(conn, first)

    assert conversations.list_all(conn)[0]["conversation_id"] == first_id


def test_list_all_empty(conn):
    assert conversations.list_all(conn) == []


def test_get_returns_turns_in_order(conn):
    conv_id, public_id = conversations.create(conn, "one")
    _document(conn)
    a1 = _turn(conn, conv_id, "one")
    a2 = _turn(conn, conv_id, "two")

    got = conversations.get(conn, public_id)

    assert [t["answer_id"] for t in got["turns"]] == [a1, a2]
    assert got["turns"][0]["question"] == "one"
    assert got["turns"][0]["abstained"] is False
    assert got["turns"][0]["citations"] == [
        {"document_id": "doc_1", "title": "Doc", "locator": "p. 1", "indexed": True}
    ]


def test_get_marks_deleted_sources_and_keeps_the_answer(conn):
    conv_id, public_id = conversations.create(conn, "q")
    _document(conn)
    _turn(conn, conv_id)
    conn.execute("DELETE FROM documents")

    (turn,) = conversations.get(conn, public_id)["turns"]

    assert turn["answer"] == "a"
    assert turn["citations"][0]["indexed"] is False


def test_get_unknown(conn):
    assert conversations.get(conn, "conv_missing") is None


def test_rename_trims_and_updates(conn):
    _, public_id = conversations.create(conn, "old")

    result = conversations.rename(conn, public_id, "  New name ")

    assert result["title"] == "New name"
    assert conversations.get(conn, public_id)["title"] == "New name"


@pytest.mark.parametrize("title", ["", "   ", "x" * 201])
def test_rename_rejects_invalid_titles(conn, title):
    _, public_id = conversations.create(conn, "old")

    with pytest.raises(conversations.InvalidTitle):
        conversations.rename(conn, public_id, title)

    assert conversations.get(conn, public_id)["title"] == "old"


def test_rename_unknown(conn):
    assert conversations.rename(conn, "conv_missing", "x") is None


def test_delete_keeps_answers_and_feedback(conn):
    conv_id, public_id = conversations.create(conn, "q")
    answer_id = _turn(conn, conv_id)
    conn.execute(
        "INSERT INTO feedback (answer_id, rating)"
        " SELECT id, 'up' FROM answers WHERE public_id = %s",
        (answer_id,),
    )

    assert conversations.delete(conn, public_id) is True
    assert conversations.delete(conn, public_id) is False

    (row,) = conn.execute(
        "SELECT a.conversation_id, count(f.id) FROM answers a"
        " LEFT JOIN feedback f ON f.answer_id = a.id GROUP BY a.conversation_id"
    ).fetchall()
    assert row == (None, 1)


RESULT = {"answer": "a", "citations": [], "abstained": False}


def test_recorder_creates_then_joins(conn):
    record = Recorder(conn)

    first_answer, conv = record("First question", RESULT, 0.3)
    before = conversations.list_all(conn)[0]["updated_at"]
    second_answer, joined = record("Second", RESULT, 0.3, conv)

    assert joined == conv and first_answer != second_answer
    assert [t["question"] for t in conversations.get(conn, conv)["turns"]] == [
        "First question",
        "Second",
    ]
    assert conversations.get(conn, conv)["title"] == "First question"
    assert conversations.list_all(conn)[0]["updated_at"] > before
    assert record.resolve(conv) and not record.resolve("conv_missing")


def test_recorder_failure_leaves_no_conversation(conn):
    with pytest.raises(KeyError):
        Recorder(conn)("q", {"answer": "a", "abstained": False}, 0.3)  # no citations

    assert conversations.list_all(conn) == []
