import psycopg
import pytest
from fastapi.testclient import TestClient

from plumbline.api.app import Recorder, app, get_conn

pytestmark = pytest.mark.db

CITED = {
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


def _ask(db, question="Refund window?", conversation_id=None, result=CITED):
    return Recorder(db)(question, result, 0.3, conversation_id)


def _document(db):
    db.execute(
        "INSERT INTO documents (public_id, content_hash, title, filename, connector)"
        " VALUES ('doc_1', repeat('a', 64), 'Doc 1', 'returns.pdf', 'pdf')"
    )


def test_list_is_empty_then_newest_first(client, db):
    assert client.get("/conversations").json() == []
    _, older = _ask(db, "older")
    _, newer = _ask(db, "newer")

    listed = client.get("/conversations").json()

    assert [c["conversation_id"] for c in listed] == [newer, older]
    assert set(listed[0]) == {"conversation_id", "title", "updated_at"}


def test_get_returns_the_turns(client, db):
    _, conv = _ask(db, "first")
    answer_id, _ = _ask(db, "second", conv)

    r = client.get(f"/conversations/{conv}")

    assert r.status_code == 200
    body = r.json()
    assert [t["question"] for t in body["turns"]] == ["first", "second"]
    assert body["turns"][1]["answer_id"] == answer_id
    assert set(body["turns"][1]) == {
        "answer_id",
        "question",
        "answer",
        "abstained",
        "citations",
    }


def test_get_unknown_is_404(client):
    r = client.get("/conversations/conv_missing")

    assert r.status_code == 404
    assert r.json() == {"detail": "Conversation not found"}


def test_deleted_source_is_marked_and_the_answer_stays(client, db):
    _document(db)
    _, conv = _ask(db)
    cited = client.get(f"/conversations/{conv}").json()["turns"][0]["citations"][0]
    assert cited["indexed"] is True

    db.execute("DELETE FROM documents")
    (turn,) = client.get(f"/conversations/{conv}").json()["turns"]

    assert turn["answer"] == "14 days [c1]"
    assert turn["citations"][0]["indexed"] is False


def test_rename_updates_the_title(client, db):
    _, conv = _ask(db)

    r = client.patch(f"/conversations/{conv}", json={"title": "  Returns  "})

    assert r.status_code == 200
    assert r.json()["title"] == "Returns"
    assert client.get("/conversations").json()[0]["title"] == "Returns"


@pytest.mark.parametrize(
    "body", [{"title": ""}, {"title": "   "}, {"title": "x" * 201}]
)
def test_rename_rejects_invalid_titles(client, db, body):
    _, conv = _ask(db, "keep me")

    assert client.patch(f"/conversations/{conv}", json=body).status_code == 422
    assert client.get(f"/conversations/{conv}").json()["title"] == "keep me"


def test_rename_without_a_title_is_422(client, db):
    _, conv = _ask(db)

    assert client.patch(f"/conversations/{conv}", json={}).status_code == 422


def test_rename_unknown_is_404(client):
    r = client.patch("/conversations/conv_missing", json={"title": "x"})

    assert r.status_code == 404


def test_delete_then_gone(client, db):
    _, conv = _ask(db)

    assert client.delete(f"/conversations/{conv}").status_code == 204
    assert client.get(f"/conversations/{conv}").status_code == 404
    assert client.get("/conversations").json() == []
    assert client.delete(f"/conversations/{conv}").status_code == 404


def test_delete_keeps_the_rated_answer(client, db):
    answer_id, conv = _ask(db)
    client.post("/feedback", json={"answer_id": answer_id, "rating": "up"})

    client.delete(f"/conversations/{conv}")

    (row,) = db.execute(
        "SELECT a.public_id, a.conversation_id, f.rating FROM answers a"
        " JOIN feedback f ON f.answer_id = a.id"
    ).fetchall()
    assert row == (answer_id, None, "up")
