# ADR 0002: Postgres + pgvector as the default store

## Status

Proposed

## Context

The default vector store was left open until the first ingestion change. The core promise
is answers we can trust, which starts with storage that never serves a
half-indexed document. v1 stores documents, their chunks, and the chunk
embeddings, and must run with one `docker compose up`.

## Decision

- One **Postgres with the pgvector extension** holds documents, chunks and
  embeddings. The schema is managed by Alembic with hand-written SQL
  migrations (`backend/migrations/`).
- A document and all of its chunks are written in **one transaction**.
  Deleting a document cascades to its chunks.
- The embedding column is typed `vector(N)`. N comes from
  `PLUMBLINE_EMBEDDING_DIM` (renamed by [ADR 0004](0004-rename-to-plumbline.md))
  when the migration runs, and has no default:
  it must match the configured embedding model.
- Retrieval uses an **exact cosine scan** in v1. An HNSW index is added in a
  later migration once corpus size makes exact scan slow. Exact scan has
  perfect recall, which keeps early eval results free of index noise.

## Alternatives

- **Embedded vector store** (Chroma, LanceDB): no extra container, but it
  needs a second store for document metadata. The two can disagree after a
  crash.
- **A separate vector database** (Qdrant, Weaviate, MongoDB vector search):
  an extra service with no v1 benefit.

## Consequences

- Self-hosters need Postgres. The bundled compose file provides it.
- **Changing the embedding model or its dimension means a new migration
  and re-ingesting every document.** Vectors from different models aren't
  comparable, so this is a real constraint, not a limitation of the schema.
- Other stores can come later behind a store interface. This ADR only sets
  the default.
