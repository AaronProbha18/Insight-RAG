# ADR 0004: YAML mirror of Plumbline's schema, with a drift gate

## Status

Accepted. Introduced by the `datamodel-system` change (issue #21), revised by
`datamodel-plumbline-mirror` (issue #28).
Architecture and technical sign-off: business tech review.

Revision note: this ADR was revised in the same PR that accepted it (issue
#28, before any release). The toolkit was trimmed to what Plumbline's mirror
needs, and the sample model, migration generator, enums, immutable tables,
uuid ids, sample rows, business rules and per-model settings were removed.

## Context

Data models were maintained by hand, so DDL, migrations and review docs
drifted. A prototype (`modelkit`) generated all three from YAML. Plumbline's
own schema, though, is Alembic's: hand-written SQL in `backend/migrations/`.
`datamodel/` therefore holds a YAML description of that schema and the tools
that check it and render it. It never touches Plumbline's database outside
the CI job below, and it never reads `backend/`.

## Decision

- **Alembic stays the source of Plumbline's schema. The YAML mirrors it.**
  `datamodel/models/plumbline/` describes the four tables, and the
  `plumbline-drift` job in `.github/workflows/datamodel.yml` fails when the
  schema generated from the YAML differs from Alembic head in columns,
  types, nullability, defaults, constraint definitions, indexes or
  triggers. It ignores object names, comments, column order and
  `alembic_version`, and labels each difference `alembic` or `yaml`. A
  Plumbline schema change updates Alembic and the YAML in the same PR.
- **Models live under `datamodel/models/<name>/`**: `domains/` holds the
  YAML, and `build/` (git-ignored) holds `schema.sql` and `explorer.html`.
  Every tool takes `--model <dir>` (relative to `datamodel/`, default
  `models/plumbline`) and `make` takes `MODEL=`.
- **System columns are injected**, never written in YAML: `id bigserial`
  (the internal primary key, never leaves the backend), and, per table's
  `system_columns`, a subset of `public_id` (unique, prefixed, opaque),
  `created_at` and `updated_at`. Foreign keys are `bigint` and point at `id`.
- **One set of conventions**, Plumbline's: no `public_id` CHECK, no
  `updated_at` trigger, no comments, and Postgres's own constraint names
  (`<t>_pkey`, `<t>_<c>_fkey`, `<t>_<cols>_key`). There is no `model.yaml`.
- **Types** are the ones Plumbline's tables use: `text`, `int`, `bigint`,
  `bool`, `timestamptz`, `jsonb`, `char(n)`, `double precision`,
  `vector(n)` and `vector({embedding_dim})`, where `{embedding_dim}` is
  filled from `PLUMBLINE_EMBEDDING_DIM` when DDL or the explorer is
  generated.
- **The explorer is one static HTML file**, data embedded as JSON, vanilla
  JS, no network. A deliberate deviation from the React + Vite default: a
  review page doesn't justify a toolchain. It is local only: built by
  `make explorer` and opened from disk, not hosted and not uploaded by CI.
- **Commands** are `make` targets calling `uv run`: `validate`, `ddl`,
  `explorer`, `test`.

## Alternatives

- For Plumbline's schema, YAML as the source with generated migrations: a
  cutover on a live schema, and `vector(N)` has no clean home in it.
- For Plumbline's schema, docs only with no gate: it drifts, which is the
  problem this ADR exists to fix.
- Keep the migration generator, the sample model and per-model settings
  from #21: code and a model Plumbline never uses.
- A modelling tool: UI, but opaque and hard to review in PRs.
- A React explorer: matches the product stack, adds a build chain.

## Consequences

- Every Plumbline schema change is written twice, in Alembic and in the
  YAML. Forgetting the YAML fails the drift job.
- The drift job is not a required check (a path-filtered required check
  would block every PR that doesn't touch these paths), so a reviewer must
  see it green.
- The drift job doesn't compare trigger function bodies, sequences or
  extensions.
- The toolkit is smaller than #21 first delivered: no migration generator,
  enums, immutable tables or per-model conventions. A feature that comes
  back needs a reason.
- Anyone reviewing the model builds the explorer themselves.
