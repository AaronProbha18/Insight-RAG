"""documents and chunks with pgvector embeddings

Revision ID: 0001
Revises:
"""

import os

from alembic import op

revision = "0001"
down_revision = None


def _embedding_dim() -> int:
    value = os.environ.get("INSIGHT_RAG_EMBEDDING_DIM", "")
    if not value.isdigit() or int(value) == 0:
        raise RuntimeError(
            "INSIGHT_RAG_EMBEDDING_DIM must be a positive integer matching the "
            f"embedding model (e.g. 768 for nomic-embed-text); got {value!r}"
        )
    return int(value)


def upgrade() -> None:
    dim = _embedding_dim()
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        CREATE TABLE documents (
          id            bigserial PRIMARY KEY,
          public_id     text UNIQUE NOT NULL,
          content_hash  char(64) UNIQUE NOT NULL,
          title         text NOT NULL,
          filename      text NOT NULL,
          connector     text NOT NULL,
          created_at    timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        f"""
        CREATE TABLE chunks (
          id           bigserial PRIMARY KEY,
          document_id  bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          ordinal      int NOT NULL,
          text         text NOT NULL,
          locator      text NOT NULL,
          embedding    vector({dim}) NOT NULL,
          UNIQUE (document_id, ordinal)
        )
        """
    )


def downgrade() -> None:
    # The vector extension stays: other objects in the database may use it.
    op.execute("DROP TABLE chunks")
    op.execute("DROP TABLE documents")
