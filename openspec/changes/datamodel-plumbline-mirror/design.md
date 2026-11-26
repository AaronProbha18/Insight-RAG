## Context

`datamodel/` (#21, [ADR 0004](../../../docs/adr/0004-yaml-data-model.md))
generates DDL from YAML, but it hard-codes conventions Plumbline doesn't
use. Plumbline's schema is Alembic head, in `backend/migrations/versions/`:

| Table | Plumbline | The toolkit today |
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
in `datamodel/models/plumbline/` (Q1a); conventions as settings with
defaults that keep the sample unchanged (Q2a); the gate as a second job in
`datamodel.yml` (Q3a).

## Decisions

### Alembic stays the source; YAML mirrors it

- **Choice**: Plumbline's schema changes go into Alembic first, then the
  same change goes into `models/plumbline/` in the same PR. CI fails when
  the two disagree.
- **Alternatives**: YAML as the source with generated migrations (B), or
  docs only with no gate (C).
- **Why**: B is a cutover on a live schema, and `vector(N)` has no clean
  home in it. C drifts, which is the problem #21 exists to fix.
- **ADR**: docs/adr/0005-plumbline-schema-mirror.md (new, Proposed)

### Models under `datamodel/models/<name>/`

- **Choice**: each model is a folder holding `domains/`, `enums/`, an
  optional `model.yaml`, an optional `migrations/` plus `model.lock.json`,
  and `build/`. Tools take `--model <dir>`, relative to `datamodel/`, with
  default `models/sample`. The Makefile has `MODEL ?= models/sample`.
  `model.py`'s `ROOT` remains the toolkit root, and a new `model_dir()`
  resolves the model.
- **Alternatives**: `backend/schema/`, which breaks ADR 0004's isolation.
- **Why**: one toolkit, several models; `datamodel/` still never reads
  `backend/`.
- **ADR**: 0005

### Conventions in `model.yaml`

```yaml
# models/plumbline/model.yaml (every key optional; defaults = today's behaviour)
id_type: bigserial            # uuid (default) | bigserial
public_id_check: false        # default true: ck_<t>_public_id
updated_at_trigger: false     # default true: tg_<t>_updated_at + set_updated_at()
constraint_names: postgres    # generated (default) | postgres: <t>_pkey, <t>_<c>_fkey, <t>_<cols>_key, <t>_<c>_check
comments: false               # default true: COMMENT ON TABLE/COLUMN
```

- Per table: `system_columns: [public_id, created_at, updated_at]`, a subset
  of the optional ones (`id` is always injected), default all three. A
  fully immutable table still drops `updated_at`.
- Foreign keys must match `id_type`: `uuid` or `bigint`.
- `constraint_names: postgres` follows Postgres's own naming. For named
  table checks (`ck_*` in YAML), Postgres names an unnamed CHECK
  `<t>_check`, `<t>_check1`... **UNVERIFIED - confirm on Postgres 18 by
  creating the Alembic schema and listing `pg_constraint`.** The gate
  ignores names anyway, so this only matters for readable DDL.
- **Alternatives**: no injection for Plumbline tables (Q2b, rejected: it
  loses the "can't forget system columns" guarantee).
- **ADR**: 0005

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
    2. `cd datamodel && make ddl MODEL=models/plumbline`, then `psql -f`
       into DB `generated` (after `CREATE EXTENSION vector`).
    3. `uv run python tools/drift.py <alembic-url> <generated-url>`.
- `tools/drift.py` builds a snapshot from `pg_catalog`, using `psql -At`
  (no Python driver, as in #21). Per table it takes:
  - columns: name, `format_type`, not-null, default;
  - constraints: `pg_get_constraintdef`, as a set (names ignored);
  - indexes: `pg_get_indexdef` with the name stripped;
  - triggers.

  It excludes `alembic_version`, prints a readable diff, and exits 1 on
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
make validate|ddl|explorer [MODEL=models/plumbline]     # default models/sample
uv run python tools/ddl.py --model models/plumbline
uv run python tools/drift.py <url-a> <url-b>            # exit 0 same, 1 drift
```

```text
datamodel/models/sample/     domains/ enums/ migrations/ model.lock.json   (moved, unchanged)
datamodel/models/plumbline/  model.yaml domains/rag/tables/{documents,chunks,answers,feedback}.yaml enums/
```

## Data

None. Plumbline's schema and migrations are unchanged; the gate only
reads databases it creates.

## Risks / Open Questions

- Every Plumbline schema change is now written twice (Alembic and YAML).
  The gate makes forgetting the YAML fail CI, but that's the cost of A.
- `feedback.reason`/`rating` checks: written in YAML as enums (ck lists)
  or as table checks. The gate compares `pg_get_constraintdef`, so the
  expression text must match Postgres's normalised form. Decided per
  column during task 3.x by running the gate.
