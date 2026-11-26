# ADR 0004: YAML data model with generated DDL and migrations

## Status

Proposed. Introduced by the `datamodel-system` change (issue #21).
Architecture and technical sign-off: business tech review.

## Context

Data models were maintained by hand, so DDL, migrations and review docs
drifted. A prototype (`modelkit`) generates all three from YAML. It lives
in `datamodel/`, separate from Plumbline's own database, which keeps its
Alembic migrations.

## Decision

- **YAML is the model.** `datamodel/domains/**` and `datamodel/enums/*`
  are the only source; `build/schema.sql`, `migrations/*.sql`,
  `model.lock.json` and `build/explorer.html` are generated. A column in
  the database but not in a `table.yaml` is a defect.
- **System columns are injected**, never written in YAML: `id` (uuid,
  internal primary key, never leaves the backend), `public_id` (unique,
  `^<prefix>_[0-9A-Za-z]+$`), `created_at`, and `updated_at` unless the
  table is fully immutable. Foreign keys point at `id`. (Plumbline's own
  tables use `bigserial`; this ADR governs `datamodel/` only.)
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

## Consequences

- We own a small generator (~700 lines) and its tests.
- Migrated column order differs from a fresh build (`ADD COLUMN`
  appends); equivalence checks ignore column order.
- Removing an enum value replaces its CHECK and fails on rows still using
  it; the migrator does not flag it as destructive.
- Anyone reviewing the model builds the explorer themselves.
