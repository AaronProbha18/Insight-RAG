# Migrating to Plumbline

The project was renamed to Plumbline ([ADR 0004](../adr/0004-rename-to-plumbline.md)).
Nothing about behaviour or the `/chat` and `/documents` API changed. The old
names stop working at once: there is no fallback.

## 1. Environment variables

Rename these in your `.env`, shell and deployment config. The suffix stays
the same.

| Old | New |
| --- | --- |
| `INSIGHT_RAG_LLM` | `PLUMBLINE_LLM` |
| `INSIGHT_RAG_EMBEDDINGS` | `PLUMBLINE_EMBEDDINGS` |
| `INSIGHT_RAG_EMBEDDING_DIM` | `PLUMBLINE_EMBEDDING_DIM` |
| `INSIGHT_RAG_JUDGE_LLM` | `PLUMBLINE_JUDGE_LLM` |
| `INSIGHT_RAG_MODEL_BASE_URL` | `PLUMBLINE_MODEL_BASE_URL` |
| `INSIGHT_RAG_ABSTAIN_THRESHOLD` | `PLUMBLINE_ABSTAIN_THRESHOLD` |
| `INSIGHT_RAG_UI_DIR` | `PLUMBLINE_UI_DIR` |

If any `INSIGHT_RAG_*` variable is still set, the app, `alembic upgrade` and
`plumbline-eval` stop with a message such as:

```text
Old environment variables are no longer read. Rename: INSIGHT_RAG_LLM -> PLUMBLINE_LLM, ...
```

`plumbline-eval` prints it after `NO GATE RESULT:` and exits with code 2.

## 2. Commands and package

| Old | New |
| --- | --- |
| `insight-rag-eval` | `plumbline-eval` |
| Python package `insight_rag` | `plumbline` |
| Distribution `insight-rag` | `plumbline-rag` |

## 3. Third-party connectors

Register connectors under the `plumbline.connectors` entry-point group and
import from `plumbline.connectors`:

```toml
[project.entry-points."plumbline.connectors"]
my-format = "my_package.connector:MyConnector"
```

A connector still registered under `insight_rag.connectors` is not loaded.
The registry logs one warning per such entry, naming `plumbline.connectors`.

## 4. Keeping an existing database

The default database user, password and name changed from `insight_rag` to
`plumbline`. A database that was created with the old defaults keeps its old
role and database name, so the new defaults cannot log in to it. You do not
need to migrate data: point Plumbline at the old values instead.

With Docker Compose, put these in `.env` next to `docker-compose.yml` before
`docker compose up`:

```text
POSTGRES_USER=insight_rag
POSTGRES_PASSWORD=insight_rag
POSTGRES_DB=insight_rag
```

Without Compose, set `DATABASE_URL` to the old connection string, for example
`postgresql://insight_rag:insight_rag@localhost:5432/insight_rag`.

Compose names the data volume after the project, which by default is the
directory name (for example `<directory>_pgdata`). Keep the directory name, or
your documents will appear to be gone because compose looks for a different
volume.

Fresh installs use the new defaults and need none of this.
