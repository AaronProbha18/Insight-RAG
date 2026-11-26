from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config

from plumbline.config import LegacyConfigError

pytestmark = pytest.mark.db

INI = Path(__file__).parents[1] / "alembic.ini"


def _query(url, sql, *args):
    with psycopg.connect(url) as conn:
        return conn.execute(sql, args).fetchall()


def test_upgrade_creates_schema(migrate, database_url):
    migrate()

    embedding_type = _query(
        database_url,
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute"
        " WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'",
    )
    uniques = _query(
        database_url,
        "SELECT pg_get_constraintdef(oid) FROM pg_constraint"
        " WHERE contype = 'u' AND conrelid IN ('documents'::regclass, 'chunks'::regclass)",
    )
    assert embedding_type == [("vector(8)",)]
    assert {u for (u,) in uniques} >= {
        "UNIQUE (content_hash)",
        "UNIQUE (public_id)",
        "UNIQUE (document_id, ordinal)",
    }


def test_chunks_cascade_with_document(migrate, database_url):
    migrate()
    with psycopg.connect(database_url) as conn:
        (doc_id,) = conn.execute(
            "INSERT INTO documents (public_id, content_hash, title, filename, connector)"
            " VALUES ('doc_x', repeat('a', 64), 't', 'f.pdf', 'pdf') RETURNING id"
        ).fetchone()
        conn.execute(
            "INSERT INTO chunks (document_id, ordinal, text, locator, embedding)"
            " VALUES (%s, 0, 'x', 'p. 1', '[0,0,0,0,0,0,0,1]')",
            (doc_id,),
        )
        conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        (count,) = conn.execute("SELECT count(*) FROM chunks").fetchone()

    assert count == 0


def test_downgrade_removes_tables(migrate, database_url):
    migrate()
    command.downgrade(Config(INI), "base")

    tables = _query(
        database_url,
        "SELECT to_regclass('documents'), to_regclass('chunks'),"
        " to_regclass('answers'), to_regclass('feedback')",
    )
    assert tables == [(None, None, None, None)]
    migrate()  # upgrades cleanly again


def _answer(conn):
    return conn.execute(
        "INSERT INTO answers (public_id, question, answer, abstained, citations,"
        " chat_model, embedding_model, abstain_threshold)"
        " VALUES ('ans_x', 'q', 'a', false, '[]', 'm', 'e', 0.5) RETURNING id"
    ).fetchone()[0]


def test_feedback_is_unique_per_answer_and_cascades(migrate, database_url):
    migrate()
    with psycopg.connect(database_url) as conn:
        answer_id = _answer(conn)
        conn.execute(
            "INSERT INTO feedback (answer_id, rating) VALUES (%s, 'up')", (answer_id,)
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            conn.execute(
                "INSERT INTO feedback (answer_id, rating) VALUES (%s, 'down')",
                (answer_id,),
            )
        conn.rollback()

    with psycopg.connect(database_url) as conn:
        conn.execute("DELETE FROM answers WHERE id = %s", (answer_id,))
        (count,) = conn.execute("SELECT count(*) FROM feedback").fetchone()
    assert count == 0


@pytest.mark.parametrize(
    "rating, reason, comment",
    [
        ("up", "wrong", None),  # reason only allowed with down
        ("down", "bogus", None),
        ("sideways", None, None),
        ("down", None, "x" * 2001),
    ],
)
def test_feedback_checks_reject(migrate, database_url, rating, reason, comment):
    migrate()
    with psycopg.connect(database_url) as conn:
        answer_id = _answer(conn)
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute(
                "INSERT INTO feedback (answer_id, rating, reason, comment)"
                " VALUES (%s, %s, %s, %s)",
                (answer_id, rating, reason, comment),
            )


def test_feedback_checks_accept_valid(migrate, database_url):
    migrate()
    with psycopg.connect(database_url) as conn:
        answer_id = _answer(conn)
        conn.execute(
            "INSERT INTO feedback (answer_id, rating, reason, comment)"
            " VALUES (%s, 'down', 'should_have_answered', %s)",
            (answer_id, "x" * 2000),
        )


def test_missing_dim_fails_clearly(migrate, monkeypatch):
    monkeypatch.delenv("PLUMBLINE_EMBEDDING_DIM")

    with pytest.raises(RuntimeError, match="PLUMBLINE_EMBEDDING_DIM"):
        migrate()


def test_legacy_variable_fails_before_the_dimension_is_read(migrate, monkeypatch):
    monkeypatch.setenv("INSIGHT_RAG_EMBEDDING_DIM", "8")

    with pytest.raises(LegacyConfigError, match="PLUMBLINE_EMBEDDING_DIM"):
        migrate()

    monkeypatch.delenv("INSIGHT_RAG_EMBEDDING_DIM")  # the fixture downgrades on exit
