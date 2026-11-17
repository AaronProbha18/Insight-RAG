## 1. Project skeleton

- [x] 1.1 Initialise git, add MIT `LICENSE`, `.gitignore`, empty `.env.example` (each later task adds the variables its code reads), README stub
- [x] 1.2 Create `backend/` uv project (`insight_rag` package, FastAPI, ruff, pytest) with `GET /health` and its test
- [x] 1.3 Add `docker-compose.yml` with Postgres + pgvector and a pytest fixture that gives each test run an isolated database
- [x] 1.4 Spike: confirm `init_chat_model` / `init_embeddings` provider strings (D6); implement `models.py` env factory with a fake-provider test; record findings in design.md D6
- [x] 1.5 Spike: choose PDF/DOCX parsing libraries - check licenses against MIT and extraction quality on one sample each; record in design.md Risks

## 2. Chat stream spike

- [x] 2.1 Spike: minimal Vite app with assistant-ui consuming a hard-coded SSE endpoint carrying text deltas + citations; pick the runtime, fix the wire format, record it in design.md D8

## 3. Connector contract

- [x] 3.1 Contract module: `CONTRACT_VERSION`, `Section`, `ParsedDocument`, `FileConnector` protocol, `ParseError`
- [x] 3.2 Registry: load the `insight_rag.connectors` entry-point group, skip version mismatches with a warning; tests with a fake connector registered via a test entry point
- [x] 3.3 Conformance kit `run_conformance()` with self-tests using a deliberately broken fake connector

## 4. Built-in connectors and fixture corpora

- [x] 4.1 Write synthetic fixture corpus (3-5 documents) as source text; generate PDF and DOCX versions into `evals/connectors/{pdf,docx}/corpus/`, recording how they were generated
- [x] 4.2 PDF connector (page locators) registered via entry point; passes conformance on the PDF corpus plus a corrupt-file fixture
- [x] 4.3 DOCX connector (heading-path locators) registered via entry point; passes conformance on the DOCX corpus plus a corrupt-file fixture

## 5. Storage and ingestion

- [x] 5.1 Alembic setup and first migration (`documents`, `chunks`, pgvector extension) per design Data section; draft ADR 0002 (Postgres + pgvector default store)
- [x] 5.2 Chunker: recursive splitting with configurable size/overlap that preserves section locators; unit tests
- [x] 5.3 Ingestion pipeline: hash -> dedupe -> parse -> chunk -> embed -> single-transaction insert; tests for idempotency and for no partial document on embedding failure
- [x] 5.4 `POST /documents` and `GET /documents` with 201/200/415/422 behaviour and `doc_<ULID>` public ids; API tests for every document-ingestion scenario

## 6. Answer graph and chat endpoint

- [x] 6.1 Retriever: cosine top-k over chunks returning text, score, document title, locator; tests with fixed fake embeddings
- [x] 6.2 LangGraph graph: retrieve -> generate | abstain with configurable threshold; prompt requiring source-only answers with chunk-id citations; tests with a fake LLM for answer, abstain, and empty-index paths
- [x] 6.3 `POST /chat` SSE endpoint in the wire format fixed in 2.1; tests for event order and no confidence field

## 7. Evaluation harness

- [x] 7.1 Golden sets: 30-50 questions per connector (>= 20% unanswerable) in `golden.yaml`, plus `evals/judge/labelled.yaml`; loader with shape validation and its test
- [x] 7.2 Judge: prompts for correctness and faithfulness, judge-validation step against labelled cases with the 0.80 agreement stop
- [x] 7.3a Scoring: per-question scoring and the four metrics (correctness, faithfulness, abstention, hit-rate@k) plus the 0.80 gates as pure functions over golden item + graph result + judge; record metric definitions in design.md D9; tests with fake judge
- [x] 7.3b Runner `insight_rag.evals` exposed as console script `insight-rag-eval`, reading data from top-level `evals/`: isolated DB per connector, ingest corpus, judge validation first, ask through the real graph, score with 7.3a, write report, exit code; `--connectors`, `--report`, `--sample`; tests with fake LLM/judge
- [ ] 7.4 Baseline run with real models: tune abstain threshold once, record threshold and results in the report and in eval-plan.md Baseline column

## 8. Chat UI

- [x] 8.1 `frontend/` Vite + React + assistant-ui app wired to `/chat` using the runtime chosen in 2.1; streamed thread renders
- [x] 8.2 Citation display: title + locator per citation, expandable cited text, none on abstention
- [x] 8.3 Upload control calling `POST /documents` with indexed / already-indexed / unsupported / unreadable feedback; keyboard and accessible-name check

## 9. Packaging

- [x] 9.1 Root multi-stage `Dockerfile` (frontend build + backend), FastAPI serves the built UI; `app` service in compose with `.env`, db health wait, migrations on start
- [x] 9.2 Smoke test against the running stack: `docker compose up --build`, then `/health`, one upload, one chat (`scripts/smoke.sh`; health verified here, full upload + chat pass pending M1 with a model)

## 10. Documentation

- [x] 10.1 README quickstart (`cp .env.example .env`, fill it, `docker compose up`; dev mode; run evals) and "write a connector" guide using the conformance kit
- [x] 10.2 Update AGENTS.md Layout and Commands with the real directories and lint/test/eval commands

## 11. CI and maintained documentation

- [x] 11.1 `.github/workflows/ci.yml` on push/PR using free `ubuntu-latest` runners: backend (uv, ruff, format check, pytest with a pgvector service container), frontend (npm ci, lint, build, Vitest), `openspec validate --all`, Docker image build without push
- [x] 11.2 `docs/architecture/overview.md`: layout, ingest / ask / eval flows, boundary rules, data model and migration rules, configuration sources; every named path exists
- [x] 11.3 `docs/guides/backend-development-guide.md`: API conventions, connectors, ingestion atomicity, graph and prompt rules, models via env, migrations, testing, Do/Don't; examples match real code
- [x] 11.4 `docs/guides/frontend-development-guide.md`: assistant-ui runtime and SSE adapter, component placement, accessibility, CSS tokens, Vitest without jsdom, Windows note, Do/Don't
- [x] 11.5 Docs maintenance: move `docs/connectors.md` to `docs/guides/connector-development-guide.md` (update links), AGENTS.md rule that behaviour changes update the overview/guides in the same PR, a docs check in rag-execute's self-review, and a CI step checking relative links in docs and READMEs
