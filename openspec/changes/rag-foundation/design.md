## Context

Greenfield. Constraints from Accepted ADRs: FastAPI with the LangGraph
graph embedded, assistant-ui frontend, entry-point
discovered ingestion connectors with a conformance kit, PDF/DOCX only, no
permissions (0001), MIT and nothing private in the repo, 80% per
connector trust bar with citations and abstention. Guiding rule:
start small - every component is the simplest one that meets the trust bar.

## Decisions

### D1. Repository layout

- **Choice**:
  ```text
  backend/                 uv project, package `insight_rag`
    src/insight_rag/
      api/                 FastAPI app and routes
      connectors/          contract (protocol, models, registry) + builtin pdf/docx
      connectors/testing/  conformance kit (importable by third-party packages)
      ingestion/           chunk, embed, store pipeline
      graph/               LangGraph answer graph
      store/               Postgres access + migrations
      evals/               eval runner + judge (console script `insight-rag-eval`)
      models.py            LLM / embedding factory from env
    tests/
  evals/                   eval DATA only: fixture corpora + golden sets per connector,
                           judge labels; reports/ (gitignored)
  frontend/                React + Vite + assistant-ui
  docker-compose.yml       postgres+pgvector
  ```
- **Alternatives**: monorepo tooling (pnpm/turbo workspaces); separate
  repos for core and connectors.
- **Why**: two apps and a folder of eval data need no workspace tooling.
  The conformance kit lives inside the core package so third-party
  connector packages import it with one dependency. Eval *code* lives in
  the backend package (same environment, one command - the pattern used by
  established open-source RAG backends); eval *data* stays top-level so
  golden sets are easy to find and edit without touching code.
- **ADR**: none

### D2. Storage: Postgres + pgvector for everything

- **Choice**: one Postgres (pgvector image) holding documents, chunks and
  embeddings. Started by `docker compose up`. Schema managed with Alembic.
- **Alternatives**: embedded vector store (Chroma, LanceDB) - zero
  infrastructure but a second store for metadata and weaker durability
  guarantees; MongoDB vector search - extra service for no v1 benefit.
- **Why**: one durable, transactional store means a document and its
  chunks are written atomically, so a crash never leaves a half-indexed
  document. Postgres is the most widely self-hostable option for an
  open-source project.
- **ADR**: draft ADR 0002 "Postgres + pgvector as default store" during
  implementation (task 1.x); other stores come later behind the store
  interface.

### D3. Connector contract v1

- **Choice**: a Python `Protocol` plus pydantic models; connectors are
  discovered from the `insight_rag.connectors` entry-point group, built-ins
  included. v1 connectors are *file* connectors: they parse uploaded bytes.
  Source-enumerating connectors (sync cursor, deletions) are deferred to the
  first non-file connector - the protocol version bump covers it.
- **Conformance fixtures** (task 3.3): `run_conformance(connector, dir)`
  uses valid files in `dir/` and files that must raise `ParseError` in
  `dir/corrupt/`, filtered by the connector's `extensions`.
- **Alternatives**: full sync protocol now (list/fetch/cursor/deletes) -
  speculative with only file inputs; plain function registry - no versioning
  or metadata for third parties.
- **Why**: the smallest contract that real PDF/DOCX connectors exercise,
  while versioned so it can grow.
- **ADR**: refines ADR 0001 (no superseding needed - 0001 left the exact
  shape to this design).

### D4. Ingestion: synchronous, idempotent

- **Choice**: `POST /documents` parses, chunks, embeds and stores within the
  request. Idempotency key = SHA-256 of the file bytes, enforced by a unique
  constraint; a duplicate returns the existing document with
  `created: false`.
- **Alternatives**: background job queue (Celery/RQ/Dagster).
- **Why**: fine for single documents in a local app; no queue to run.
  Ceiling: large files block the request. Upgrade path: move the same
  pipeline function behind a job queue when upload size or volume demands.
- **ADR**: none

### D5. Chunking

- **Choice**: recursive character splitting (`langchain-text-splitters`, chosen in task 5.2), default ~1000 characters with
  ~150 overlap, configurable. Each chunk keeps its page number (PDF) or
  heading path (DOCX) for citations.
- **Alternatives**: semantic chunking, layout-aware parsing.
- **Why**: simplest baseline; the eval harness tells us when it falls short.
- **ADR**: none

### D6. Models: provider-agnostic via environment

- **Choice**: `INSIGHT_RAG_LLM` and `INSIGHT_RAG_EMBEDDINGS` hold
  `provider:model` strings resolved through LangChain's provider-agnostic
  initializers (`backend/src/insight_rag/models.py`). Embedding dimension is
  read from config and fixed in the vector column; changing embedding model
  requires re-ingestion (documented).
- **Spike findings (task 1.4, langchain 1.5.0 / langchain-core 1.8.1)**:
  - `langchain.chat_models.init_chat_model` splits on the first `:` when the
    prefix is a known provider, so `ollama:llama3.1:8b` gives model
    `llama3.1:8b`. Chat providers include anthropic, azure_openai, bedrock,
    cohere, deepseek, fireworks, google_genai, groq, huggingface, mistralai,
    ollama, openai, openrouter, together, xai.
  - `langchain.embeddings.init_embeddings` also splits on the first `:`, but
    supports fewer providers: azure_ai, azure_openai, bedrock, cohere,
    google_genai, google_vertexai, huggingface, mistralai, ollama, openai.
    Groq, together and fireworks are chat-only here.
  - Each provider needs its partner package (`langchain-<provider>`). The
    core install ships none; users add the one they pick.
  - `DeterministicFakeEmbedding` needs numpy (not a dependency), so the tests
    stub embeddings instead.
- **Reference hardware**: Apple M1 with 16 GB, running Ollama natively
  (Metal) with `llama3.1:8b` and `nomic-embed-text`; the full eval runs here.
  A minimum dev machine without a usable GPU (16 GB, integrated Intel GPU)
  runs every unit test with fakes. The `llm` marker test skips without
  Ollama. Hardware changes only `.env`, never code.
- **Alternatives**: LiteLLM proxy; hard-coded single provider.
- **Why**: no extra service; any LangChain-supported provider, including
  local models, works.
- **ADR**: none

### D7. Answer graph

- **Choice**: two-node LangGraph graph with a conditional edge.
  ```text
  START -> retrieve(top_k) --(best score >= threshold)--> generate -> END
                           \--(otherwise)--------------> abstain  -> END
  ```
  `generate` is prompted to answer only from the retrieved chunks, cite them
  by id, and say it cannot answer when they are insufficient. The abstain
  threshold is calibrated from the first eval run, not guessed.
- **Alternatives**: query rewriting, grading/retry loops, reranker.
- **Why**: smallest graph that implements cited answers and abstention;
  additions must earn their place by moving eval metrics.
- **ADR**: none

### D8. Chat stream API and assistant-ui runtime

- **Choice**: `POST /chat` returns a streamed response (Server-Sent Events)
  carrying text deltas, then citations, then a done event.
- **Runtime (task 2.1 spike, `@assistant-ui/react` 0.15.36)**:
  `useLocalRuntime(adapter)` with our own `ChatModelAdapter`. Its
  `run({messages, abortSignal})` is an async generator yielding
  `{content: [{type: "text", text}], metadata: {custom: {...}}}`. The
  adapter `fetch`es `/chat` and parses SSE itself (~15 lines, no SSE
  library). Citations travel in `metadata.custom.citations`, not
  assistant-ui `source` parts. Those only carry url/title/mediaType, with no
  locator or cited text. The UI reads them in the
  `ThreadPrimitive.Messages` children render function (`message.metadata.custom`).
  Rejected: the AI-SDK data-stream runtime, because it would tie the backend
  wire format to the Vercel AI SDK protocol.
- **Wire format** (request `{"messages": [{"role", "content"}]}`; response
  `text/event-stream`, named events, one JSON object per `data:` line):

  | Event | Data | When |
  | --- | --- | --- |
  | `delta` | `{"text": str}` | 0..n, answer text in order |
  | `citations` | `{"citations": [{"id", "document_id", "title", "locator", "text"}]}` | once, after deltas, only for an answer |
  | `abstain` | `{"message": str}` | instead of deltas + citations |
  | `error` | `{"message": str}` | on failure, then the stream ends |
  | `done` | `{}` | last event of a successful stream |

  No confidence field (answers are trusted via citations and evals, not a self-reported score).
- **Streaming mode (task 6.3, option A)**: the graph runs to completion and
  its verified result is then sent as `delta` events, so unverified text is
  never shown. Known limitation: the first `delta` arrives only after the
  full answer is generated. Move to live token streaming (with retraction)
  only if 7.4 latency shows the wait hurts.
- **Alternatives**: LangGraph Server protocol (rejected: extra platform components for self-hosters);
  non-streaming JSON.
- **Why**: streaming is table stakes for chat UX; deciding the format by
  spike avoids inventing an API the library cannot consume.
- **ADR**: none until the format stabilises; it becomes a public contract.

### D9. Evaluation

- **Choice**: minimal in-repo eval runner (`insight_rag.evals`, run as
  `insight-rag-eval`, reading data from top-level `evals/`): for each
  connector, ingest its fixture corpus into a throwaway database, ask each
  golden question through the real graph, score with an LLM judge, apply
  the eval-plan.md thresholds, exit non-zero on failure. Judge checked first against
  ~10 human-labelled cases per metric.
- **Metric definitions** (task 7.3a, `evals/scoring.py`):

  | Metric | Over | Hit when | Gated |
  | --- | --- | --- | --- |
  | correctness | answerable questions | answered **and** judge agrees with the reference; abstaining counts as wrong | >= 0.80 |
  | faithfulness | answered questions | judge finds every claim supported by the **cited** chunks | >= 0.80 |
  | abstention | unanswerable questions | the graph abstained | >= 0.80 |
  | hit-rate@k | answerable questions | a retrieved chunk matches a supporting (document, locator) | tracked |

  An empty denominator scores 0.0 (fails closed).
- **Alternatives**: Ragas, DeepEval. UNVERIFIED whether either runs
  without extra services or accounts; both add sizeable dependency trees.
- **Why**: four metrics fit in a small file and avoid tying the trust bar to
  a third-party library's metric definitions. Revisit if metrics multiply.
- **ADR**: none

### D10. Packaging: one app image + database

- **Choice**: one multi-stage root `Dockerfile` - a Node stage builds
  `frontend/`, a Python (uv) stage installs `backend/` and copies the built
  UI in; FastAPI serves the API and the static UI on one port. Compose
  services: `db` (pgvector, named volume) and `app` (this image, reads
  `.env`, waits for `db` health, runs migrations on start, then serves).
  `docker compose up` starts everything. For development, `docker compose
  up db` plus local `fastapi dev` / `vite dev` still works.
- **Alternatives**: separate backend and frontend (nginx) images - lets
  each scale independently, but adds a second Dockerfile and CORS/proxy
  config with no v1 benefit.
- **Why**: the simplest thing an open-source user can run: fill `.env`,
  one command. Ceiling: UI and API ship and scale together. Upgrade path:
  split into two images when a real deployment needs it.
- **ADR**: none

## Interfaces

```python
# insight_rag/connectors/contract.py
from typing import Protocol, ClassVar
from pydantic import BaseModel

CONTRACT_VERSION = 1

class Section(BaseModel):
    text: str
    locator: str            # e.g. "page 3" or "Introduction > Scope"

class ParsedDocument(BaseModel):
    title: str
    sections: list[Section]
    metadata: dict[str, str] = {}

class FileConnector(Protocol):
    name: ClassVar[str]                  # "pdf", "docx"
    contract_version: ClassVar[int]      # == CONTRACT_VERSION
    media_types: ClassVar[tuple[str, ...]]
    extensions: ClassVar[tuple[str, ...]]

    def parse(self, data: bytes, filename: str) -> ParsedDocument: ...

# pyproject.toml of any connector package
# [project.entry-points."insight_rag.connectors"]
# pdf = "insight_rag.connectors.builtin.pdf:PdfConnector"

# insight_rag/connectors/testing/conformance.py
def run_conformance(connector: FileConnector, fixtures_dir: Path) -> None:
    """Raise AssertionError on any contract violation:
    version match, non-empty sections, every section has a locator,
    deterministic output for identical bytes, clean error on corrupt input."""
```

```text
POST /documents        multipart file -> 201 {document_id, title, connector, chunks, created}
                       200 same body with created=false on duplicate
                       415 unsupported type, 422 unparseable file
GET  /documents        -> [{document_id, title, connector, created_at}]
POST /chat             {messages:[{role, content}]} -> SSE stream (format per D8)
GET  /health           -> {status}
```

Public ids: `document_id` is an opaque string (`doc_` + ULID); internal
integer keys never leave the backend.

## Data

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id            bigserial PRIMARY KEY,
  public_id     text UNIQUE NOT NULL,
  content_hash  char(64) UNIQUE NOT NULL,   -- sha256 hex, idempotency key
  title         text NOT NULL,
  filename      text NOT NULL,
  connector     text NOT NULL,
  created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
  id           bigserial PRIMARY KEY,
  document_id  bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ordinal      int NOT NULL,
  text         text NOT NULL,
  locator      text NOT NULL,
  embedding    vector(:dim) NOT NULL,       -- dim from config at migration time
  UNIQUE (document_id, ordinal)
);
-- ANN index (hnsw, cosine) added once corpus size makes exact scan slow;
-- exact scan is correct and fast enough for v1 volumes.
```

Document + chunks are inserted in one transaction.

## Risks / Open Questions

- Parsing libraries (task 1.5 spike, one synthetic sample each):
  - PDF: `pypdf` 7.1.0 (BSD-3-Clause). Per-page `extract_text()` gives
    correct page locators and non-ASCII text (`café`). Quirk: table cells
    come out one per line (`Item\nDays\nShoes\n30`), so row structure is
    lost, though the values stay adjacent. Fallback if evals show table
    questions failing: `pdfplumber` 0.11.10 (MIT, pulls `pdfminer.six`, MIT),
    which keeps rows (`Item Days\nShoes 30`).
  - DOCX: `python-docx` 1.2.0 (MIT). Paragraph style names give heading
    levels (`Heading 1`/`Heading 2`) for heading-path locators. Quirk:
    tables are not in `document.paragraphs`. The connector must walk
    `document.element.body` in order (`p`/`tbl`) so table text lands under
    its heading.
  - Rejected: PyMuPDF (AGPL, incompatible with MIT distribution).
  - Known limits: no OCR (scanned PDFs yield no text and must fail as
    unreadable); bullet glyphs may surface as control characters (`\x7f`)
    and should be stripped by the connector.
- Risk: LLM-judge scores drift by judge model; the judge model is pinned in
  eval config and recorded in every eval report.
- OPEN QUESTION: which LLM/embedding provider the eval baseline is recorded
  against. Assumption: whatever the maintainer configures; the report
  records it, and the 80% bar applies to that configuration.
