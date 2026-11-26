## Context

`datamodel/` (#21, [ADR 0004](../../../docs/adr/0004-yaml-data-model.md))
generates DDL from YAML, but it hard-codes conventions Plumbline doesn't
use, and it carries features that only a synthetic sample model used.
Plumbline's schema is Alembic head, in `backend/migrations/versions/`:

| Table | Plumbline | The toolkit as delivered in #21 |
| --- | --- | --- |
| all | `id bigserial PRIMARY KEY` | `id uuid DEFAULT gen_random_uuid()` |
| `documents`, `answers` | `public_id text UNIQUE NOT NULL`, no CHECK | `public_id` plus `ck_<t>_public_id` |
| `chunks`, `feedback` | no `public_id` | always injected |
| `chunks`, `feedback` | no `created_at` | always injected |
| `documents`, `chunks`, `answers` | no `updated_at` | injected unless fully immutable |
| `feedback` | `updated_at`, no trigger | trigger `tg_<t>_updated_at` |
| `chunks` | `embedding vector(N)`, N from `PLUMBLINE_EMBEDDING_DIM` | no `vector` type |
| `documents`, `answers` | `char(64)`, `double precision` | not allowed types |
| all | default names: `documents_pkey`, `chunks_document_id_fkey`, `feedback_rating_check` | `pk_`/`fk_`/`uq_`/`ck_` names |
| all | no comments | `COMMENT ON` every table and column |

Decisions taken in triage (owner): a mirror with a drift gate (A); the YAML
in `datamodel/models/plumbline/` (Q1a); the gate as a second job in
`datamodel.yml` (Q3a). Revised during implementation (owner): conventions
were first settings with defaults that keep a sample unchanged (Q2a); the
owner then chose to remove the sample and everything only it used, so
Plumbline's conventions became the toolkit's only behaviour.

## Decisions

### Alembic stays the source; YAML mirrors it

- **Choice**: Plumbline's schema changes go into Alembic first, then the
  same change goes into `models/plumbline/` in the same PR. CI fails when
  the two disagree.
- **Alternatives**: YAML as the source with generated migrations (B), or
  docs only with no gate (C).
- **Why**: B is a cutover on a live schema, and `vector(N)` has no clean
  home in it. C drifts, which is the problem #21 exists to fix.
- **ADR**: docs/adr/0004-yaml-data-model.md (rewritten in place)

### Models under `datamodel/models/<name>/`

- **Choice**: a model is a folder holding `domains/` and `build/`. Tools
  take `--model <dir>`, relative to `datamodel/`, with default
  `models/plumbline`. The Makefile has `MODEL ?= models/plumbline`.
  `model.py`'s `ROOT` remains the toolkit root, and `model_dir()` resolves
  the model. `--model` stays so tests can point the tools at a temporary
  model.
- **Alternatives**: `backend/schema/`, which breaks ADR 0004's isolation.
- **Why**: `datamodel/` still never reads `backend/`.
- **ADR**: 0004

### Plumbline's conventions are the toolkit's behaviour

- The generated DDL has `id bigserial`, `bigint` foreign keys, no
  `public_id` CHECK, no `updated_at` trigger, no `COMMENT ON`, and
  Postgres-style names (`<t>_pkey`, `<t>_<c>_fkey`, `<t>_<cols>_key`).
  Named table checks (`checks:` in YAML) keep the name written in YAML.
  There is no `model.yaml` and no per-model switch.
- Per table: `system_columns: [public_id, created_at, updated_at]`, a subset
  of the optional ones (`id` is always injected), default all three.
  `public_id_prefix` is required only when `public_id` is injected.
- Foreign keys must be `bigint`.
- Postgres's own name for an unnamed CHECK is unverified and doesn't matter:
  the gate ignores names.
- **Alternatives**: keeping the conventions as per-model settings with the
  sample as default (the first design, task 2.1), rejected by the owner:
  no other model exists, and the settings would be dead code.
- **ADR**: 0004

### Trim the toolkit to what Plumbline uses

- **Choice**: remove `models/sample/`, `migrate.py`, `model.lock.json` and
  `migrations/` (with `make migrate` and `make check`), enums, immutable
  tables, uuid ids, `sample_rows` and `business_rules`. Keep `schema`,
  `model`, `validate`, `ddl`, `drift`, the explorer, and the review metadata
  (`description`, `pii`, `status`, `open_questions`). The tests build a small
  fixture inside `tests/` instead of reading a model folder.
- **Alternatives**:
  - keep the sample and the generator as a test fixture (A): code and a
    model Plumbline never uses;
  - drop the explorer too: rejected, it is in use and shows Plumbline's
    model.
- **Why**: Plumbline owns its migrations in Alembic, has no enums or
  immutable tables, and uses `bigserial` ids. Unused features are untested
  code and extra review surface.
- **ADR**: 0004. It is Accepted in the same PR, and not released before, so
  it is rewritten in place with a revision note instead of superseded.
- **Order**: the sample uses the features being removed, so it goes first
  (6.1), then migrations (6.2), then the features (6.3), then the settings
  (6.4).

### New types

- **Choice**: allow `char(n)`, `double precision`, `vector(n)` and
  `vector({embedding_dim})`. `ddl.py` and `explorer.py` replace
  `{embedding_dim}` from the `PLUMBLINE_EMBEDDING_DIM` env var; if it's
  unset, they fail with a message naming the variable. `validate.py` doesn't
  need the variable.
- **Why**: mirrors how migration 0001 reads the same variable.

### Drift gate

- **Choice**: a `plumbline-drift` job in `datamodel.yml`, on paths
  `datamodel/**`, `backend/migrations/**` and the workflow itself.
  - Service: `pgvector/pgvector:pg18` (same as backend CI).
  - `PLUMBLINE_EMBEDDING_DIM=8`.
  - Steps:
    1. `cd backend && uv sync --locked && uv run alembic upgrade head`
       against DB `alembic`.
    2. `cd datamodel && make ddl`, then `psql -f` into DB `generated`
       (after `CREATE EXTENSION vector`).
    3. `uv run python tools/drift.py <alembic-url> <generated-url>`, with
       side labels `alembic` and `yaml`.
- `tools/drift.py` builds a snapshot from `pg_catalog`, using `psql -At`
  (no Python driver, as in #21). Per table it takes:
  - columns: name, `format_type`, not-null, default;
  - constraints: `pg_get_constraintdef`, as a set (names ignored);
  - indexes: `pg_get_indexdef` with the name stripped;
  - triggers.

  It excludes `alembic_version`, ignores column order, prints a readable
  diff labelled by side (`only in alembic`, `only in yaml`), and exits 1 on
  any difference.
- **Alternatives**:
  - raw `pg_dump -s` diff: always differs on names and comments;
  - a backend test: makes `backend/` call into `datamodel/`.
- **Why**: compares what matters for behaviour, and keeps the two folders
  separate.
- Verified in triage: `backend/migrations/env.py` reads `DATABASE_URL`
  (via `plumbline.db.database_url`) and runs `check_legacy_env`; migration
  0001 needs `PLUMBLINE_EMBEDDING_DIM`. No other env is needed.

## Interfaces

```bash
make validate|ddl|explorer [MODEL=models/plumbline]     # default models/plumbline
uv run python tools/ddl.py --model models/plumbline
uv run python tools/drift.py <url-a> <url-b>            # exit 0 same, 1 drift
```

```text
datamodel/models/plumbline/  domains/rag/domain.yaml, domains/rag/tables/{documents,chunks,answers,feedback}.yaml
```

## Data

None. Plumbline's schema and migrations are unchanged; the gate only
reads databases it creates.

## Risks / Open Questions

- Every Plumbline schema change is now written twice (Alembic and YAML).
  The gate makes forgetting the YAML fail CI, but that's the cost of A.
- The toolkit is smaller than #21 delivered: there is no migration
  generator, enum or immutable-table support. Anything that needs them
  later has to bring them back with a reason.
- `datamodel-system` (the main spec) still describes the removed features
  until the delta spec in task 6.6 is archived.
- Whether the YAML check expressions render like Alembic's inline CHECKs was
  decided by running the gate (task 4.2): `feedback` checks are written
  without `IS NULL OR`, as Alembic wrote them.
