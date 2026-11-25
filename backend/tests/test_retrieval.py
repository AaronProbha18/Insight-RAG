import math

import psycopg
import pytest
from fake_embeddings import FixedEmbeddings

from plumbline.retrieval import retrieve

pytestmark = pytest.mark.db


def _unit(degrees: float) -> list[float]:
    """8-dim vector at `degrees` from the x axis in the first two dims."""
    r = math.radians(degrees)
    return [math.cos(r), math.sin(r)] + [0.0] * 6


@pytest.fixture
def conn(migrate, database_url):
    migrate()
    with psycopg.connect(database_url, autocommit=True) as c:
        yield c


def _seed(conn, *chunks: tuple[str, str, list[float]]):
    (doc,) = conn.execute(
        "INSERT INTO documents (public_id, content_hash, title, filename, connector)"
        " VALUES ('doc_A', repeat('a', 64), 'Manual', 'm.pdf', 'pdf') RETURNING id"
    ).fetchone()
    for i, (text, locator, vec) in enumerate(chunks):
        conn.execute(
            "INSERT INTO chunks (document_id, ordinal, text, locator, embedding)"
            " VALUES (%s, %s, %s, %s, %s::vector)",
            (doc, i, text, locator, str(vec)),
        )


def test_returns_nearest_first(conn):
    _seed(
        conn,
        ("far", "p. 1", _unit(80)),
        ("near", "p. 2", _unit(10)),
        ("mid", "p. 3", _unit(45)),
    )

    results = retrieve(conn, FixedEmbeddings(_unit(0)), "q")

    assert [r.text for r in results] == ["near", "mid", "far"]
    assert (results[0].document_id, results[0].title, results[0].locator) == (
        "doc_A",
        "Manual",
        "p. 2",
    )


def test_score_is_cosine_similarity(conn):
    _seed(conn, ("same", "p. 1", _unit(0)), ("orthogonal", "p. 2", _unit(90)))

    scores = {r.text: r.score for r in retrieve(conn, FixedEmbeddings(_unit(0)), "q")}

    assert scores["same"] == pytest.approx(1.0, abs=1e-6)
    assert scores["orthogonal"] == pytest.approx(0.0, abs=1e-6)


def test_top_k_limits_results(conn):
    _seed(conn, *[(f"c{i}", "p. 1", _unit(i * 10)) for i in range(6)])

    assert len(retrieve(conn, FixedEmbeddings(_unit(0)), "q", top_k=2)) == 2


def test_empty_index_returns_empty(conn):
    assert retrieve(conn, FixedEmbeddings(_unit(0)), "q") == []
