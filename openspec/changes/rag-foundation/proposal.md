## Why

There is no application yet. This change delivers the smallest end-to-end
RAG that proves the project's core promise: upload a PDF or Word
document, ask a question, get a cited answer - and *measure* that answers
meet the 80% trust bar, per connector.

## What Changes

- Python backend (FastAPI + embedded LangGraph graph) with a
  Postgres + pgvector store started via Docker Compose.
- Connector contract v1 (ADR 0001): ingestion protocol, entry-point
  discovery, and a reusable conformance test kit.
- Two built-in ingestion connectors: PDF and DOCX, each with a synthetic
  fixture corpus and golden set.
- Ingestion pipeline: parse -> chunk -> embed -> store, idempotent by
  content hash (re-uploading the same file is a no-op).
- Answer graph: retrieve -> answer with citations, or abstain when nothing
  relevant is retrieved.
- Eval harness enforcing the 80% gate per connector (eval-plan.md), runnable locally
  with one command.
- React + Vite frontend using assistant-ui: chat with streamed answers,
  citation display, and file upload.
- Provider-agnostic model configuration: LLM and embedding model selected
  by environment variables.
- One-command run: a single multi-stage `Dockerfile` (API + built chat UI)
  and a `docker compose up` that starts the app and its database once
  `.env` is filled in.

## Non-Goals

- Authentication, users, multi-tenancy, per-document permissions.
- Query connectors (Postgres, Snowflake), Excel/CSV, any connector beyond
  PDF/DOCX.
- Query rewriting, retrieval grading loops, reranking, hybrid search.
- OCR for scanned PDFs, tables/images inside documents.
- Production deployment (Kubernetes, cloud hosting, TLS, scaling); CI pipeline.
- Conversation history persistence across server restarts.

## Capabilities

### New

- `connector-contract`: protocol, discovery, and conformance kit for ingestion connectors.
- `document-ingestion`: upload, parse via connector, chunk, embed, store, idempotency.
- `cited-answering`: question in, streamed answer with citations or an abstention out.
- `answer-evaluation`: golden sets, scoring, and the per-connector 80% gate.
- `chat-ui`: assistant-ui chat with upload and citations.

### Modified

- none

## Risk

| Area | Yes/No | Note |
| --- | --- | --- |
| Answer quality (retrieval, ranking, chunking, embeddings, prompts, graph) | Yes | Establishes the first baseline for every metric |
| Connector / ingestion | Yes | Defines contract v1 and the first two connectors |
| Auth / document ACLs | No | Explicitly out of scope (ADR 0001); single-user local app |
| Public API or stream contract | Yes | Upload API, chat stream format, and connector protocol become public contracts |
| Persistence / migration | Yes | First schema: documents, chunks, vectors |
| New external dependency or service | Yes | Postgres + pgvector, an LLM provider, an embedding provider |

## Impact

New: `backend/`, `frontend/`, `evals/`, `Dockerfile`, `docker-compose.yml`,
`.env.example`, `LICENSE` (MIT), README quickstart. AGENTS.md gains the
real layout and commands.
