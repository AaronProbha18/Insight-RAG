import secrets
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from psycopg import sql

from plumbline import db


@pytest.fixture(scope="session")
def database_url():
    """A throwaway database with pgvector, created for this test run and dropped after."""
    admin_url = db.database_url()
    try:
        admin = psycopg.connect(admin_url, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as exc:
        pytest.skip(
            f"Postgres unreachable at DATABASE_URL ({exc}); run `docker compose up -d db`"
        )

    name = f"insight_rag_test_{secrets.token_hex(4)}"
    admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    url = psycopg.conninfo.make_conninfo(admin_url, dbname=name)
    try:
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        yield url
    finally:
        admin.execute(
            sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name))
        )
        admin.close()


@pytest.fixture
def migrate(database_url, monkeypatch):
    """Run alembic on the isolated database (8-dim vectors); always ends at base."""
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("PLUMBLINE_EMBEDDING_DIM", "8")
    cfg = Config(Path(__file__).parents[1] / "alembic.ini")
    yield lambda rev="head": command.upgrade(cfg, rev)
    command.downgrade(cfg, "base")
