# AGENTS.md

Tool-neutral source of truth for anyone (human or AI agent) working in this
repo. Tool-specific notes live elsewhere (`CLAUDE.md` for Claude Code).

## Purpose

insight-rag-v2 is a retrieval-augmented generation (RAG) application. Content
enters through **connectors**, is indexed, and is answered over by a
LangGraph agent, behind a chat UI.

## Stack

| Layer | Choice | ADR |
| --- | --- | --- |
| Agent / orchestration | Python, LangGraph (embedded in FastAPI) | - |
| API | FastAPI | - |
| Frontend | React + Vite, assistant-ui | - |
| Connectors | Pluggable, entry-point discovered | 0001 |
| Store | Postgres + pgvector | 0002 (Proposed) |
| Open source | MIT | - |
| Answer trust bar | >= 80% per connector, cited, abstains | - |

Versions are not pinned here; check the lockfiles.

## Layout

```text
backend/                   Python package `insight_rag` (uv project)
  src/insight_rag/
    api/app.py             FastAPI routes (/health, /documents, /chat); serves the built UI in Docker
    connectors/            contract.py (v1), registry.py (entry points),
                           builtin/ (pdf, docx), testing/ (conformance kit)
    ingestion.py           hash -> dedupe -> parse -> chunk -> embed -> atomic insert
    chunking.py            section-preserving splitter
    retrieval.py           cosine top-k over chunks
    graph.py               LangGraph answer graph: retrieve -> generate | abstain
    models.py              chat / embedding / judge models from env
    db.py                  DATABASE_URL with the local default
    evals/                 golden + label loaders, judge, scoring, runner (insight-rag-eval)
  migrations/              Alembic, hand-written SQL
  tests/                   pytest; `db` marker needs Postgres, `llm` needs Ollama
frontend/                  Vite + React + assistant-ui
  src/chat/                SSE parser, assistant-ui adapter, citations
  src/upload/              upload control
evals/                     eval DATA only: corpus-src/, connectors/<name>/{corpus,golden.yaml},
                           judge/labelled.yaml, reports/ (gitignored)
scripts/smoke.sh           full-stack smoke test
docs/adr/                  architecture decision records - why things are the way they are
docs/connectors.md         guide: write a connector
openspec/                  specs (living), changes (in flight), schemas (our workflow)
.claude/                   Claude Code skills, commands, hooks
Dockerfile                 one image: built UI + API
docker-compose.yml         db (pgvector) + app
```

## Workflow

```text
idea -> /opsx:explore -> /opsx:propose (pick tier) -> per tasks.md checkbox:
rag-plan -> rag-execute -> ... -> rag-verify -> PR (human merges) -> /opsx:archive
```

1. **Tier.** `rag-standard` is the default (proposal -> design -> specs +
   eval-plan -> tasks -> verification). `rag-rapid` (proposal -> tasks ->
   verification) only when nothing touches retrieval, prompts, connectors,
   ingestion, auth/ACL, a public contract, or persistence. The proposal's
   Risk table justifies the tier; never downgrade to lighten the process.
2. **No change needed** for repo-internal work: CI, lint, `.claude/`,
   `openspec/schemas/`, dependency bumps, ADR-only edits. Ship as a
   `chore/` PR.
3. **One checkbox per loop.** A task that does not fit one plan/execute
   loop is split in `tasks.md`, not absorbed.
4. **Evals are evidence.** A change with Answer quality = yes ships an
   `eval-plan.md`, and its `verification.md` shows results against the
   baseline. A regression is a FAIL.
5. **ADRs** record decisions that outlive one change (`docs/adr/`,
   `NNNN-kebab-title.md`, Status: Proposed / Accepted / Superseded by NNNN).
   `design.md` links to them. An Accepted ADR is followed or superseded,
   never silently contradicted.
6. **Humans merge.** Agents may draft commits and PRs; they never merge,
   force-push, or bypass checks.

## Data and safety rules

- No real user, customer, or production data in prompts, fixtures, eval
  sets, logs, or commits. Synthetic or public data only.
- No secrets in the repo. Reference environment variable names; commit
  `.env.example`, never `.env`.
- Open source: no internal URLs, org names, ticket keys, or
  non-public data anywhere in the repo, including commit messages.
- Unverified library or vendor behaviour is marked UNVERIFIED with how to
  confirm it, never asserted.

## Commands

```bash
# backend (from backend/)
uv sync --extra ollama             # install, incl. the Ollama provider
uv run ruff check . && uv run ruff format --check .
uv run pytest                      # db tests skip if Postgres is down

# frontend (from frontend/; on Windows use an upper-case drive letter, D:\...)
npm run lint && npm run build
npm test

# stack (from repo root)
docker compose up -d db            # Postgres + pgvector for dev and tests
docker compose up --build          # full app on :8000
sh scripts/smoke.sh                # health, one upload, one chat

# evals (from backend/; needs model env vars, see README)
uv run insight-rag-eval [--connectors pdf] [--sample N]

# fixture corpus (from repo root, after editing evals/corpus-src/)
uv run --no-project --with reportlab --with python-docx python evals/generate_corpus.py

# openspec
openspec list                      # active changes
openspec schema validate <name>    # after editing a schema
openspec validate --all            # before opening a PR
```
