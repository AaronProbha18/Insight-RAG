Issue: #28

## Why

`datamodel/` (#21) only holds a synthetic sample model. Plumbline's four
tables live in hand-written Alembic SQL, so the validator and explorer say
nothing about the schema the app actually uses, and nothing notices when
docs and the database disagree. The toolkit also carries features that only
the synthetic sample uses.

## What Changes

- **The Plumbline model** at `datamodel/models/plumbline/`: `documents`,
  `chunks`, `answers` and `feedback`, described in YAML to match Alembic
  head. The tools and the Makefile take `--model <dir>`, with default
  `models/plumbline`.
- **Per-table `system_columns`** to opt out of `public_id`, `created_at` or
  `updated_at` (`id` is always injected).
- **New column types:** `vector(n)` and `vector({embedding_dim})` (filled
  from `PLUMBLINE_EMBEDDING_DIM` at generation time), `char(n)` and
  `double precision`.
- **Plumbline's conventions are the toolkit's only behaviour:** `bigserial`
  ids with `bigint` foreign keys, no `public_id` CHECK, no `updated_at`
  trigger, no comments, Postgres-style constraint names.
- **Everything Plumbline doesn't use is removed:** `models/sample/`,
  `migrate.py`, `model.lock.json` and `migrations/` (with `make migrate` and
  `make check`), enums, immutable tables, uuid ids, `sample_rows` and
  `business_rules`. The tests move to a small fixture built inside `tests/`.
- **Kept:** the explorer, and the review metadata `description`, `pii`,
  `status` and `open_questions`.
- **Drift gate:** a second job in `.github/workflows/datamodel.yml`, which
  also triggers on `backend/migrations/**`. It runs `alembic upgrade head`
  on one database and the generated DDL on another (pgvector image), then
  compares a structural snapshot of the two. The snapshot covers columns,
  types, nullability, defaults, and constraint and index definitions. It
  ignores names, comments and `alembic_version`, and its output labels the
  two sides `alembic` and `yaml`.
- **ADR 0004** is rewritten in place: Alembic stays the source of truth for
  Plumbline's schema, YAML mirrors it, gated by CI, and the toolkit is
  limited to what that needs. The ADR is accepted in this same PR and not
  released before, so there's no earlier Accepted text to supersede.

## Non-Goals

- YAML as the source of truth for Plumbline, generated migrations, or any
  cutover from Alembic (option B, rejected in triage).
- Changing Plumbline's schema or its Alembic revisions.
- Keeping the sample model, the migration generator, or support for several
  conventions.
- Making the drift gate a required check on `main`.

## Capabilities

### New

- `datamodel-plumbline-mirror`: model selection, Plumbline's conventions,
  the Plumbline model, the drift gate, and the limited toolkit.

### Modified

- `datamodel-system`: its Migrations, validator, DDL generation and
  explorer requirements shrink (no migrations, enums, immutable tables,
  uuid ids or sample rows). The delta spec is written in task 6.6.

## Risk

| Area | Yes/No | Note |
| --- | --- | --- |
| Answer quality (retrieval, ranking, chunking, embeddings, prompts, graph) | No | Reads the schema only; no runtime code changes. |
| Connector / ingestion | No | |
| Auth / document ACLs | No | |
| Public API or stream contract | No | `--model` is an internal tooling contract, recorded in ADR 0004. |
| Persistence / migration | Yes | A CI gate reads Plumbline's migrated schema. Plumbline's schema and migrations are not changed. |
| New external dependency or service | No | The gate reuses the pgvector image and the backend's existing dependencies. |

## Impact

`datamodel/` (tools, Makefile, tests, README; the sample and the
migration generator are removed), `.github/workflows/datamodel.yml`,
`docs/adr/0004-yaml-data-model.md`, `docs/architecture/overview.md`,
`AGENTS.md`. Backend runtime code is untouched. Architecture sign-off:
business tech review.
