# ADR 0004: YAML data model with generated DDL and migrations

## Status

Accepted. Introduced by the `datamodel-system` change (issue #21), extended
by `datamodel-plumbline-mirror` (issue #28).
Architecture and technical sign-off: business tech review.

## Context

Data models were maintained by hand, so DDL, migrations and review docs
drifted. A prototype (`modelkit`) generates all three from YAML. It lives
in `datamodel/` and never touches Plumbline's own database or reads
`backend/`. Plumbline's own schema keeps its Alembic migrations, and
`datamodel/` also holds a YAML mirror of it so the two can't drift unseen.

## Decision

- **YAML is the model.** `datamodel/models/<name>/domains/**` and
  `datamodel/models/<name>/enums/*` are the only source; `build/schema.sql`,
  `migrations/*.sql`, `model.lock.json` and `build/explorer.html` are
  generated. For a model whose YAML is the source (`models/sample`), a
  column in the database but not in a `table.yaml` is a defect.
- **Models live under `datamodel/models/<name>/`.** Every tool takes
  `--model <dir>` (relative to `datamodel/`, default `models/sample`) and
  `make` takes `MODEL=`. Each model has its own `build/`, and, when it
  generates migrations, its own `migrations/` and `model.lock.json`.
- **System columns are injected**, never written in YAML: `id` (internal
  primary key, never leaves the backend), then `public_id` (unique,
  `^<prefix>_[0-9A-Za-z]+$`), `created_at`, and `updated_at` (unless the
  table is fully immutable). A table's `system_columns` can list a subset
  of the three optional ones; the default is all three. Foreign keys point
  at `id`.
- **Conventions are per-model settings** in an optional `model.yaml`, with
  defaults that keep the sample unchanged: `id_type` (`uuid` or
  `bigserial`), `public_id_check`, `updated_at_trigger`, `comments` and
  `constraint_names` (`generated` or Postgres's own names).
- **Types** beyond the basics: `char(n)`, `double precision`, `vector(n)`
  and `vector({embedding_dim})`, where `{embedding_dim}` is filled from
  `PLUMBLINE_EMBEDDING_DIM` when DDL or the explorer is generated.
- **Alembic stays the source for Plumbline's schema.** `models/plumbline/`
  mirrors it, and the `plumbline-drift` job in
  `.github/workflows/datamodel.yml` fails when the schema generated from the
  YAML differs from Alembic head in columns, types, nullability, defaults,
  constraint definitions, indexes or triggers. It ignores object names,
  comments, column order and `alembic_version`. A Plumbline schema change
  updates Alembic and the YAML in the same PR.
- **Enums are CHECK constraints**, not Postgres ENUM types.
- **Migrations are a snapshot diff** of generated objects against
  `model.lock.json`, committed together. Destructive changes are refused
  without `--allow-destructive`, which is recorded in the file. No renames,
  data or down migrations; revisit (e.g. Atlas) if renames become common.
- **Explorer is one static HTML file**, data embedded as JSON, vanilla JS,
  no network. A deliberate deviation from the React + Vite default: a
  review page doesn't justify a toolchain.
- **Explorer is local only**: built by `make explorer` and opened from
  disk; not hosted and not uploaded by CI.
- **Commands** are `make` targets calling `uv run`.

## Alternatives

- SQL/Alembic first, document after: standard tooling, docs rot.
- A modelling tool: UI, but opaque and hard to review in PRs.
- Postgres ENUM types: stronger typing, but removing a value is hard.
- Alembic autogenerate or Atlas: understand renames, but need a live DB.
- React explorer: matches the product stack, adds a build chain.
- For Plumbline's schema, YAML as the source with generated migrations: a
  cutover on a live schema, and `vector(N)` has no clean home in it.
- For Plumbline's schema, docs only with no gate: it drifts, which is the
  problem this ADR exists to fix.

## Consequences

- We own a small generator and its tests.
- Migrated column order differs from a fresh build (`ADD COLUMN`
  appends); equivalence checks ignore column order.
- Removing an enum value replaces its CHECK and fails on rows still using
  it; the migrator does not flag it as destructive.
- Anyone reviewing the model builds the explorer themselves.
- Every Plumbline schema change is written twice, in Alembic and in the
  YAML. Forgetting the YAML fails the drift job.
- The drift job is not a required check (a path-filtered required check
  would block every PR that doesn't touch these paths), so a reviewer must
  see it green.
- The drift job doesn't compare trigger function bodies, sequences or
  extensions.
