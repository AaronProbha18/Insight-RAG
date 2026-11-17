from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config

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
        "SELECT to_regclass('documents'), to_regclass('chunks')",
    )
    assert tables == [(None, None)]
    migrate()  # upgrades cleanly again


def test_missing_dim_fails_clearly(migrate, monkeypatch):
    monkeypatch.delenv("INSIGHT_RAG_EMBEDDING_DIM")

    with pytest.raises(RuntimeError, match="INSIGHT_RAG_EMBEDDING_DIM"):
        migrate()
