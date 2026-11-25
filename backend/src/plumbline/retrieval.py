"""Exact cosine top-k search over stored chunks."""

from dataclasses import dataclass

import psycopg


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int  # internal; never sent to clients
    document_id: str  # public doc_<ULID>
    title: str
    locator: str
    text: str
    score: float  # cosine similarity, higher is closer


def retrieve(conn: psycopg.Connection, embeddings, question: str, top_k: int = 5):
    query = str(list(embeddings.embed_query(question)))
    rows = conn.execute(
        "SELECT c.id, d.public_id, d.title, c.locator, c.text,"
        " 1 - (c.embedding <=> %s::vector) AS score"
        " FROM chunks c JOIN documents d ON d.id = c.document_id"
        " ORDER BY c.embedding <=> %s::vector, c.id LIMIT %s",
        (query, query, top_k),
    ).fetchall()
    return [RetrievedChunk(*row) for row in rows]
