# insight-rag

Open-source retrieval-augmented generation: plug in a source through a
connector, ask questions, and get answers that cite where they came from -
or say the answer is not in your sources. Answer quality is measured, not
assumed: every supported connector must clear an 80% accuracy bar on its
evaluation set.

**Status:** pre-alpha. PDF and Word (DOCX) documents, local models via
Ollama or any provider LangChain supports.

## How trust works

Every answer cites the passages it relies on, with the document title and a
page number or heading path, and you can open each one. When the retrieved
passages don't support an answer, insight-rag says so rather than guess.
Each connector is gated by an evaluation run: correctness, faithfulness and
abstention on unanswerable questions must each reach 0.80, scored by an LLM
judge that is itself checked against human labels first.

## Quickstart (Docker)

You need Docker and a model provider. The reference setup is a 16 GB Apple
M1 running [Ollama](https://ollama.com) natively, so it uses the GPU.

1. **Models** (check the Ollama library for current tags):

   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```

2. **Configure:**

   ```bash
   cp .env.example .env
   ```

   | Variable | Required | Example | Purpose |
   | --- | --- | --- | --- |
   | `INSIGHT_RAG_LLM` | yes | `ollama:llama3.1:8b` | chat model, as `provider:model` |
   | `INSIGHT_RAG_EMBEDDINGS` | yes | `ollama:nomic-embed-text` | embedding model |
   | `INSIGHT_RAG_EMBEDDING_DIM` | yes | `768` | vector size of that embedding model; fixed at first start |
   | `INSIGHT_RAG_MODEL_BASE_URL` | in Docker | `http://host.docker.internal:11434` | where the container reaches Ollama on your machine |
   | `INSIGHT_RAG_JUDGE_LLM` | for evals | `ollama:llama3.1:8b` | eval judge; deliberately separate from the chat model |
   | `INSIGHT_RAG_ABSTAIN_THRESHOLD` | no | `0.3` | minimum retrieval similarity before answering |
   | `DATABASE_URL` | no | `postgresql://insight_rag:insight_rag@localhost:5432/insight_rag` | defaults to the compose `db` service on localhost; compose sets its own for the app |
   | `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | no | `insight_rag` | database credentials; the defaults are for local use only |

   (`INSIGHT_RAG_UI_DIR` is set by the Docker image to serve the built UI;
   you don't set it.)

3. **Run:**

   ```bash
   docker compose up --build
   ```

   Open <http://localhost:8000>, upload a PDF or DOCX, and ask a question.
   To check the whole stack from a terminal, run `sh scripts/smoke.sh`.

> **Changing the embedding model** changes the vector size, and the
> database column is fixed at first start. Run `docker compose down -v`
> (this deletes indexed documents), update `INSIGHT_RAG_EMBEDDING_DIM`, and
> start again.

Other providers: install their LangChain package by building with
`--build-arg PROVIDER=<extra>`. Only `ollama` is wired up today; add an extra in
`backend/pyproject.toml` for others.

## Development

```bash
docker compose up -d db                     # Postgres + pgvector only

cd backend
uv sync --extra ollama
uv run alembic upgrade head                 # needs INSIGHT_RAG_EMBEDDING_DIM
uv run fastapi dev src/insight_rag/api/app.py   # API on :8000

cd frontend
npm install
npm run dev                                 # UI on :5173, proxies the API
```

### Tests

```bash
cd backend && uv run ruff check . && uv run pytest   # DB tests skip if Postgres is down
cd frontend && npm run lint && npm test
```

On Windows, run the frontend tests from a path with an **upper-case drive
letter** (`D:\...`). From `d:\...`, Vitest loads itself twice and every
suite fails with "failed to find the current suite".

### Evaluations

`uv run insight-rag-eval` scores every connector against its golden set and
exits 0 (pass), 1 (a connector is below 0.80) or 2 (judge not validated, so
no result). See [evals/README.md](evals/README.md).

## Contributing

Work starts as a GitHub issue: pick the **Feature or change** or **Bug**
form. A maintainer triages it into a planned change (`/rag-triage`), then
it's built task by task into a draft PR (`/rag-quick`). The issue is the
source of truth throughout. The loops, labels and rules are in
[AGENTS.md](AGENTS.md#workflow); review uses [REVIEW.md](REVIEW.md).
Use synthetic data only, never real documents or personal data.

## Learn more

- [docs/architecture/overview.md](docs/architecture/overview.md) - how the system fits together: flows, boundary rules, data model
- [docs/guides/backend-development-guide.md](docs/guides/backend-development-guide.md) - rules for backend code
- [docs/guides/frontend-development-guide.md](docs/guides/frontend-development-guide.md) - rules for the chat UI
- [docs/guides/connector-development-guide.md](docs/guides/connector-development-guide.md) - write your own connector
- [AGENTS.md](AGENTS.md) - stack, workflow, contribution rules
- [docs/adr/](docs/adr/) - architecture decisions
- [LICENSE](LICENSE) - MIT
