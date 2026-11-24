"""Ingest one file: hash -> dedupe -> parse -> chunk -> embed -> atomic insert."""

import hashlib
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg import errors

from plumbline.chunking import chunk
from plumbline.connectors.contract import FileConnector

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


class UnsupportedFileType(Exception):
    """No registered connector handles this file's extension."""


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    title: str
    connector: str
    chunks: int
    created: bool


def new_document_id() -> str:
    """`doc_` + ULID: 48-bit ms timestamp + 80 random bits, Crockford base32."""
    value = (time.time_ns() // 1_000_000) << 80 | secrets.randbits(80)
    ulid = "".join(_CROCKFORD[(value >> shift) & 31] for shift in range(125, -1, -5))
    return f"doc_{ulid}"


def _existing(conn: psycopg.Connection, content_hash: str) -> IngestResult | None:
    row = conn.execute(
        "SELECT d.public_id, d.title, d.connector, count(c.id)"
        " FROM documents d LEFT JOIN chunks c ON c.document_id = d.id"
        " WHERE d.content_hash = %s GROUP BY d.id",
        (content_hash,),
    ).fetchone()
    return IngestResult(*row, created=False) if row else None


def ingest(
    conn: psycopg.Connection,
    data: bytes,
    filename: str,
    connectors: dict[str, FileConnector],
    embeddings,
) -> IngestResult:
    content_hash = hashlib.sha256(data).hexdigest()
    if existing := _existing(conn, content_hash):
        return existing

    suffix = Path(filename).suffix.lower()
    connector = next((c for c in connectors.values() if suffix in c.extensions), None)
    if connector is None:
        supported = sorted(e for c in connectors.values() for e in c.extensions)
        raise UnsupportedFileType(
            f"unsupported file type {suffix or '(none)'!r}; supported: {', '.join(supported)}"
        )

    doc = connector.parse(data, filename)
    chunks = chunk(doc)
    # Embed before any write: a provider failure must leave nothing stored.
    vectors = embeddings.embed_documents([c.text for c in chunks])

    document_id = new_document_id()
    try:
        with conn.transaction():
            (doc_pk,) = conn.execute(
                "INSERT INTO documents (public_id, content_hash, title, filename, connector)"
                " VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (document_id, content_hash, doc.title, filename, connector.name),
            ).fetchone()
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (document_id, ordinal, text, locator, embedding)"
                    " VALUES (%s, %s, %s, %s, %s::vector)",
                    [
                        (doc_pk, c.ordinal, c.text, c.locator, str(list(v)))
                        for c, v in zip(chunks, vectors, strict=True)
                    ],
                )
    except errors.UniqueViolation:
        # A concurrent upload of the same bytes won the race; return its document.
        if existing := _existing(conn, content_hash):
            return existing
        raise
    return IngestResult(
        document_id, doc.title, connector.name, len(chunks), created=True
    )
