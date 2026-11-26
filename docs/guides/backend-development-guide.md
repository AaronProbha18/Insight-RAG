# Backend development guide

Rules for writing backend code in Plumbline, for people and AI agents
alike. Every rule describes how the code already works; where a rule is
enforced by a test, the test is named. For the system's shape, read
[the architecture overview](../architecture/overview.md) first. For coding
style in general, follow the guidelines in [CLAUDE.md](../../CLAUDE.md):
think first, keep it simple, make surgical changes, and verify.

## Contents

1. [Code organisation](#1-code-organisation)
2. [API conventions](#2-api-conventions)
3. [Errors](#3-errors)
4. [Ingestion invariants](#4-ingestion-invariants)
5. [Answer graph and prompts](#5-answer-graph-and-prompts)
6. [Models](#6-models)
7. [Database and migrations](#7-database-and-migrations)
8. [Testing](#8-testing)
9. [Style](#9-style)
10. [Quick reference](#10-quick-reference)

## 1. Code organisation

**1.1 Routes are thin.** A route validates input, calls one module, and
maps that module's errors to status codes. Logic lives in `ingestion.py`,
`retrieval.py`, `graph.py` and so on, where it can be tested without HTTP.

```python
# ✅ api/app.py: the route only translates
    except UnsupportedFileType as exc:
        raise HTTPException(415, str(exc)) from exc
    except ParseError as exc:
        raise HTTPException(422, str(exc)) from exc
```

**1.2 Two composition roots.** Only `api/app.py` and `evals/runner.py`
wire together the database, models, connectors and graph. Every other
module receives what it needs as arguments.

**1.3 Inject dependencies; don't import them.** The graph gets a retriever
*function* and an LLM, so its tests need no database and no model:

```python
# ✅ api/app.py
def get_retriever(conn: Conn, embeddings: Annotated[object, Depends(get_embeddings)]):
    return lambda question, top_k: retrieve(conn, embeddings, question, top_k)
```

```python
# ❌ graph.py reaching for infrastructure
import psycopg
```

*Enforced by* `tests/test_boundaries.py`.

**1.4 Configuration has owners.** Environment variables are read only in
`models.py`, `db.py`, `config.py` (the legacy-variable check) and the two
composition roots. *Enforced by*
`tests/test_boundaries.py`.

**1.5 Sync routes, one connection per request.** Routes are plain `def`,
so FastAPI runs them in a threadpool, and `get_conn` opens one psycopg
connection per request with a 5-second connect timeout. That's simple and
enough for v1. A pool (`psycopg_pool`) comes when load demands it; the
code marks this with a `ponytail:` comment.

## 2. API conventions

**2.1 Status codes mean one thing each.**

| Code | When |
| --- | --- |
| 201 | a new document was indexed |
| 200 | the same bytes were already indexed (`created: false`), so uploads are idempotent |
| 204 | `DELETE /documents/{document_id}` removed the document and its chunks |
| 404 | `DELETE /documents/{document_id}` for an unknown or already-deleted id; the message is always "Document not found" |
| 204 | (also) `POST /feedback` stored the rating, or replaced the earlier one for that answer |
| 404 | (also) `POST /feedback` for an unknown `answer_id`; the message is always "Answer not found" |
| 200 | `GET /quality` always; no usable summary is `state: "not_measured"`, never an error |
| 200 | (also) `GET /connectors` always; a connector that failed to load or declares another contract version is not listed |
| 415 | no connector handles the file type; the message lists the supported types |
| 422 | the connector can't read the file (`ParseError`), or the request body is invalid (for `/feedback`: unknown rating or reason, a reason on thumbs up, a comment over 2000 characters) |
| 500 | anything else; the client never sees internal details |

**2.2 Public ids only.** Clients see `document_id` (`doc_<ULID>`) and
`answer_id` (`ans_<ULID>`). Internal integer ids, `RetrievedChunk.chunk_id`
and a source file's `filename` (kept on the recorded answer for the
feedback export) never leave the backend: `/chat` strips `filename` from the
`citations` event.

**2.3 No confidence numbers.** Responses never carry a model-reported
confidence or a retrieval score. Trust comes from citations and evals.
*Enforced by* `test_no_confidence_field_anywhere` in `tests/test_chat_api.py`.

**2.4 `/chat` is a contract.** Its SSE events (`delta`, `citations`,
`abstain`, `error`, `done`) are fixed in design D8, and
`frontend/src/chat/adapter.ts` consumes them. Changing one means changing
the other **in the same PR**. `done` carries `{"answer_id": "ans_<ULID>"}`
for answers and abstentions, and is absent on the error path. An answer is
recorded (`feedback.record_answer`) before it streams; if recording fails
the client gets `error`, never an answer it cannot rate.

**2.5 `/quality` is an allowlist.** `quality.load_quality` reads the eval
runner's summary file (`PLUMBLINE_EVAL_SUMMARY`, default
`evals/reports/latest.summary.json`) and builds the response from named
fields only: the three gated metrics, PASS/FAIL and `failed` as the runner
wrote them, the run date, model ids, threshold and a `config_matches` check
of chat model, embedding model and threshold. `host`, `git_commit`, `report`,
`reason`, latency and every other summary field are never returned. A missing,
malformed, sampled or exit-2 summary is `not_measured`. *Enforced by*
`test_no_internal_fields_leak` in `tests/test_quality.py` and
`tests/test_quality_api.py`.

**2.6 `/connectors` is an allowlist.** The route lists the registry that
`get_connectors` returns, sorted by name, and builds each entry from three
named fields: `name`, `extensions` and `contract_version`. Module paths,
entry-point names and every other attribute stay out. A connector the
registry skipped (it failed to load, or declares another contract version)
is not listed. The names equal the keys of `/quality`, which the Connectors
page uses to join the two. *Enforced by* `tests/test_connectors_api.py`.

## 3. Errors

**3.1 Domain exceptions, mapped at the edge.** Modules raise
`UnsupportedFileType`, `ParseError` or `JudgeNotValidated`. Routes and the
eval CLI turn them into status codes and exit codes.

**3.2 A broad `except` belongs only at a trust boundary**, where it turns
an unknown failure into a contract outcome:

```python
# ✅ connectors/builtin/pdf.py: a malformed upload becomes a 422, never a 500
        except Exception as exc:  # any malformed upload is a ParseError, never a 500
            raise ParseError(f"unreadable PDF: {exc}") from exc
```

```python
# ❌ swallowing errors inside business logic
try:
    vectors = embeddings.embed_documents(texts)
except Exception:
    vectors = []
```

**3.3 Log details; send generic messages.** The chat route logs the
exception and sends the client only a fixed message:

```python
# ✅ api/app.py
    except Exception:
        log.exception("answer generation failed")
        events = [_sse("error", {"message": "answer generation failed"})]
```

## 4. Ingestion invariants

These are the guarantees behind "never serve a half-indexed document".
Don't break them without an eval-backed reason and a design note.

| Invariant | Where | Proven by |
| --- | --- | --- |
| Identical bytes are stored once (SHA-256 key) | `ingestion.ingest` | `test_same_bytes_twice_is_idempotent` |
| Embed **before** any write | `ingestion.ingest` | `test_embedding_failure_stores_nothing` |
| Document + chunks in **one transaction** | `ingestion.ingest` | `test_insert_failure_rolls_back` |
| A chunk never crosses a section, so it keeps an exact locator | `chunking.chunk` | `test_never_crosses_sections` |

```python
# ✅ ingestion.py
    # Embed before any write: a provider failure must leave nothing stored.
    vectors = embeddings.embed_documents([c.text for c in chunks])
```

## 5. Answer graph and prompts

**5.1 An answer counts only with valid citations.**

```python
# ✅ graph.py
        # An answer that says it cannot answer, or cites nothing, cannot be trusted.
        if NOT_IN_SOURCES in answer or not cited:
```

**5.2 Three ways to abstain:** an empty index, a best score below the
threshold (the LLM isn't called), or a model reply that is
`NOT_IN_SOURCES` or uncited. An abstention carries no citations.

**5.3 Prompt and threshold changes are answer-quality changes.** Editing
`PROMPT`, `DEFAULT_THRESHOLD`, `top_k` or chunk sizes changes behaviour.
Run `plumbline-eval`, compare with the baseline, and record the result
(AGENTS.md workflow rule 4). Tuning stops at the eval plan's pass rule:
change parameters, never golden answers.

**5.4 Stream only verified text.** `/chat` runs the graph to completion
before streaming (D8, option A), so an answer later withdrawn is never
shown.

## 6. Models

**6.1 Models come from `provider:model` strings** in `PLUMBLINE_LLM`,
`PLUMBLINE_EMBEDDINGS` and `PLUMBLINE_JUDGE_LLM`, created by
LangChain's `init_chat_model` / `init_embeddings` in `models.py`. Never
import a provider class directly.

**6.2 A new provider is a `pyproject.toml` extra**, like `ollama`, so the
core stays provider-free. The Docker image picks it with
`--build-arg PROVIDER=<extra>`.

**6.3 The judge is separate on purpose.** Changing the chat model must not
silently change the judge that grades it.

**6.4 Remote endpoints use `PLUMBLINE_MODEL_BASE_URL`**, forwarded as
`base_url`:

```python
# ✅ models.py
    base_url = os.environ.get("PLUMBLINE_MODEL_BASE_URL")
    return {"base_url": base_url} if base_url else {}
```

## 7. Database and migrations

**7.1 Schema changes are new Alembic revisions** in `migrations/versions/`,
written as plain SQL through `op.execute`. There are no ORM models and no
autogenerate.

**7.2 Never edit a revision that has been applied anywhere.** Add a new
one.

**7.3 The vector dimension is fixed at migration time** from
`PLUMBLINE_EMBEDDING_DIM`, with no default. A different embedding model
means a new migration and re-ingestion (ADR 0002).

**7.4 One default database location:** `db.database_url()`. Don't
hard-code a URL anywhere else.

**7.5 A new revision also updates `datamodel/models/plumbline/`** in the
same PR. The `plumbline-drift` CI job applies Alembic head and the generated
YAML schema to two databases and fails when they differ (ADR 0004).

## 8. Testing

**8.1 Fakes, not mocks.** Small hand-written stand-ins that behave like
the real thing: `tests/fake_embeddings.py` (`FakeEmbeddings`,
`FailingEmbeddings`, `FixedEmbeddings`), `tests/fake_connectors.py`, and
recording or scripted LLMs that keep the prompts they receive.

**8.1a `/chat` tests inject a fake recorder** through the `get_recorder`
dependency (`FakeRecorder` in `tests/test_chat_api.py`), so they need no
database; `tests/test_feedback.py` checks the stored row against real tables.

**8.2 Database tests are marked `db`** and use the `database_url` and
`migrate` fixtures from `tests/conftest.py`. Each run gets its own
throwaway database. Without Postgres they **skip** with a hint; they never
touch your dev data.

**8.3 Tests that need a real model are marked `llm`** and skip without
Ollama. CI runs everything except these.

**8.4 Test the failure paths.** Every ingestion invariant above has a test
in which something fails. New code that can fail needs the same.

**8.5 Every spec scenario maps to a test.** `rag-verify` checks this
before a change can pass.

## 9. Style

- `ruff check` (including import sorting) and `ruff format` are both
  enforced in CI.
- Prefer the standard library, then already-installed dependencies. A new
  dependency needs a reason in the plan.
- Mark a deliberate shortcut with a `ponytail:` comment that names its
  ceiling and the upgrade path.
- No speculative abstractions: no interface with a single implementation,
  and no configuration for a value nobody sets.

## 10. Quick reference

| Do | Don't |
| --- | --- |
| Keep routes thin; put logic in modules | Put business logic in a route handler |
| Inject retriever, LLM and embeddings | Import `psycopg` or `fastapi` into `graph.py` |
| Raise `ParseError`, `UnsupportedFileType` | Return error strings or `None` |
| Embed before writing; write in one transaction | Insert the document, then embed |
| Expose `doc_<ULID>` only | Return internal ids or retrieval scores |
| Log exceptions; send a generic message | Send stack traces or hostnames to clients |
| Add a provider as a `pyproject` extra | Import a provider SDK in core code |
| Add a new migration revision | Edit an applied one |
| Update `datamodel/models/plumbline/` with the revision | Merge a revision the YAML doesn't mirror |
| Run evals after prompt or threshold changes | Edit golden answers to pass |
| Use fakes and the `db` / `llm` markers | Call real models in unit tests |
