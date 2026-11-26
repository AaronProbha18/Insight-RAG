# datamodel

A YAML description of Plumbline's database schema and the tools around it:
a validator, a DDL generator, a drift check against Alembic, and a static
explorer page. Alembic (`backend/migrations/`) is the source of the schema;
`models/plumbline/` mirrors it, and CI fails when the two disagree. Why it
works this way: [ADR 0004](../docs/adr/0004-yaml-data-model.md).

This folder is a separate uv project, and nothing here imports from
`backend/`.

## Layout

```text
models/<m>/domains/<d>/domain.yaml        a domain: layer, load order, depends_on
models/<m>/domains/<d>/tables/<t>.yaml    a table: columns, checks, unique, indexes, open questions
models/<m>/build/                         schema.sql, explorer.html (generated, git-ignored)
tools/                                    model, schema, validate, ddl, drift, explorer (+ explorer.template.html)
tests/                                    pytest; fixtures/mini is a tiny model for the tests
```

`models/plumbline/` is the only model. `tests/fixtures/mini/` exists only so
the tests can edit YAML without touching Plumbline's.

## Commands

Run from `datamodel/`, after `uv sync`. Every target works on one model:
`MODEL=models/<m>` (default `models/plumbline`), or `--model <dir>` on the
tools. Without `make` (e.g. Windows), use the right-hand column; it needs
`PYTHONPATH=tools` (`$env:PYTHONPATH="tools"` in PowerShell).

| make | Without make |
| --- | --- |
| `make validate` | `uv run python tools/validate.py` |
| `make ddl` | `uv run python tools/ddl.py` -> `build/schema.sql` |
| `make explorer` | `uv run python tools/explorer.py` -> `build/explorer.html` |
| `make test` | `uv run pytest -q -rs` |
| `make all` | validate, ddl, explorer |
| (no target) | `uv run python tools/drift.py <url-a> <url-b> [--label-a alembic --label-b yaml]`: exit 0 same structure, 1 drift, 2 `psql` failed |

`ddl` and `explorer` need `PLUMBLINE_EMBEDDING_DIM` (a positive integer)
because Plumbline's `chunks.embedding` is `vector({embedding_dim})`; they
fail naming it if it's unset. `validate` doesn't need it.

`make validate test` is the `datamodel` CI job (`.github/workflows/datamodel.yml`).

### Postgres tests

`tests/test_postgres.py` loads the fixture's generated DDL into a real
database; `tests/test_drift.py` runs `tools/drift.py` on pairs of databases.
They run when `DATAMODEL_TEST_DATABASE_URL` points at a maintenance database
(e.g. `postgresql://postgres:pw@localhost:5432/postgres`) and `psql` is on
PATH; otherwise they are skipped. Each test creates and drops its own
temporary databases. They use plain types, so any recent Postgres works: CI
runs them on 16, and the drift job uses pgvector on 18.

## A table YAML

- `table`, `domain`, `status` (`draft` or `published`), `description`.
- `public_id_prefix`: 2 to 6 lowercase letters, required only when the table
  gets `public_id`.
- `system_columns`: a subset of `public_id`, `created_at` and `updated_at`
  (default: all three). `id bigserial` is always injected. None of these is
  written in `columns`.
- `columns`: `name`, `type`, `description`, and optionally `nullable`,
  `default` (raw SQL), `unique`, `pii`, and `references` (`table.id`) with
  `on_delete`. A foreign key column must be `bigint`.
- Types: `text`, `int`, `bigint`, `bool`, `timestamptz`, `jsonb`,
  `char(n)`, `double precision`, `vector(n)` and `vector({embedding_dim})`.
- `checks` (`name`, `expr`), `unique` (column groups), `indexes`,
  `open_questions`.

Generated DDL uses Postgres's own constraint names (`<t>_pkey`,
`<t>_<c>_fkey`, `<t>_<cols>_key`) and has no comments, no `public_id` CHECK
and no triggers, because that is what Plumbline's Alembic SQL has.

## Plumbline's model and the drift rule

`models/plumbline/` mirrors the Alembic schema in `backend/migrations/`.
**Change Alembic and the YAML in the same PR.** The `plumbline-drift` job in
`.github/workflows/datamodel.yml` fails otherwise. It applies Alembic head to
one database, loads the DDL generated from `models/plumbline` into another
(`PLUMBLINE_EMBEDDING_DIM=8`, `CREATE EXTENSION vector` first), and runs
`tools/drift.py` on the two. Columns (type, nullability, default),
constraint definitions, indexes and triggers are compared; object names,
comments, column order and `alembic_version` are ignored. Trigger function
bodies, sequences and extensions are not compared. Each difference says
whether it is only in `alembic` or only in `yaml`. The job is not a required
check, so look for it green on the PR.

To run the same comparison locally, point `drift.py` at two Postgres
databases with pgvector, one built by `alembic upgrade head` and one from
`PLUMBLINE_EMBEDDING_DIM=8 make ddl`.

## Changing the model

1. Write the Alembic revision first (see the backend development guide).
2. Edit the YAML in `models/plumbline/` to match, in the same PR.
3. `make validate`: errors name the table and column.
4. `PLUMBLINE_EMBEDDING_DIM=8 make ddl`, then run the drift check above. CI
   runs the same check.

## Explorer

`make explorer` writes one self-contained `build/explorer.html`. Open it
straight from disk: it needs no server or network, and it isn't hosted or
uploaded anywhere.

## Rules

- Identifiers longer than 63 bytes are rejected; Postgres would silently
  truncate them.
- **No real data.** Fixtures and tests use obviously synthetic values.
