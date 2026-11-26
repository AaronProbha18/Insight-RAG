Issue: #28

## Why

`datamodel/` (#21) only holds a synthetic sample model. Plumbline's four
tables live in hand-written Alembic SQL, so the validator and explorer say
nothing about the schema the app actually uses, and nothing notices when
docs and the database disagree.

## What Changes

- **Several models.** The sample moves to `datamodel/models/sample/`.
  Plumbline's model is added at `datamodel/models/plumbline/`. The tools and
  the Makefile take `--model <dir>` (`MODEL ?= models/sample`).
- **Per-model conventions** in an optional `model.yaml`, with defaults that
  leave the sample unchanged:
  - `id_type: uuid | bigserial`;
  - whether to generate the `public_id` CHECK and the `updated_at` trigger;
  - Postgres-default constraint names instead of `pk_`/`fk_`/`uq_`/`ck_`.
- **Per-table `system_columns`** to opt out of `public_id`, `created_at` or
  `updated_at` (`id` is always injected).
- **New column types:** `vector(n)` and `vector({embedding_dim})` (filled
  from `PLUMBLINE_EMBEDDING_DIM` at generation time), `char(n)` and
  `double precision`.
- **The Plumbline model:** `documents`, `chunks`, `answers` and `feedback`,
  described in YAML to match Alembic head. Synthetic sample rows only.
- **Drift gate:** a second job in `.github/workflows/datamodel.yml`, which
  also triggers on `backend/migrations/**`. It runs `alembic upgrade head`
  on one database and the generated DDL on another (pgvector image), then
  compares a structural snapshot of the two. The snapshot covers columns,
  types, nullability, defaults, and constraint and index definitions. It
  ignores names, comments and `alembic_version`.
- **ADR 0005** records the mirror decision: Alembic stays the source of
  truth for Plumbline's schema, and YAML mirrors it, gated by CI.

## Non-Goals

- YAML as the source of truth for Plumbline, generated migrations, or any
  cutover from Alembic (option B, rejected in triage).
- Changing Plumbline's schema or its Alembic revisions.
- `migrations/` or `model.lock.json` for the Plumbline model: Alembic owns
  migrations.
- Making the drift gate a required check on `main`.

## Capabilities

### New

- `datamodel-plumbline-mirror`: model selection, per-model conventions, the
  Plumbline model, and the drift gate.

### Modified

- none. `datamodel-system` behaviour is unchanged for the sample model; its
  defaults are kept.

## Risk

| Area | Yes/No | Note |
| --- | --- | --- |
| Answer quality (retrieval, ranking, chunking, embeddings, prompts, graph) | No | Reads the schema only; no runtime code changes. |
| Connector / ingestion | No | |
| Auth / document ACLs | No | |
| Public API or stream contract | No | `--model` and `model.yaml` are internal tooling contracts, recorded in ADR 0005. |
| Persistence / migration | Yes | A CI gate reads Plumbline's migrated schema. Plumbline's schema and migrations are not changed. |
| New external dependency or service | No | The gate reuses the pgvector image and the backend's existing dependencies. |

## Impact

`datamodel/` (tools, Makefile, tests, README; the sample moves under
`models/`), `.github/workflows/datamodel.yml`,
`docs/adr/0005-plumbline-schema-mirror.md`,
`docs/architecture/overview.md`, `AGENTS.md`. Backend runtime code is
untouched. Architecture sign-off: business tech review.
