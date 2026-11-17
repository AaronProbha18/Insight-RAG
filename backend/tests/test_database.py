import psycopg
import pytest

pytestmark = pytest.mark.db


def test_isolated_database_has_pgvector(database_url):
    with psycopg.connect(database_url) as conn:
        extensions = {
            row[0] for row in conn.execute("SELECT extname FROM pg_extension")
        }
        roundtrip = conn.execute("SELECT '[1,2,3]'::vector::text").fetchone()[0]

    assert "vector" in extensions
    assert roundtrip == "[1,2,3]"


def test_isolated_database_is_not_the_main_one(database_url):
    dbname = psycopg.conninfo.conninfo_to_dict(database_url)["dbname"]

    assert dbname.startswith("insight_rag_test_")
