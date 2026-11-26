# datamodel

A YAML data model and the tools that check it and generate from it: the
DDL, ordered migrations, and a static explorer page. The YAML is the only
source; everything else is generated. Why it works this way:
[ADR 0004](../docs/adr/0004-yaml-data-model.md).

This is **not** Plumbline's own schema. Plumbline's tables stay in Alembic
(`backend/migrations/`). This folder is a separate uv project, and nothing
here imports from `backend/`.

## Layout

```text
domains/<d>/domain.yaml          a domain: layer, load order, depends_on
domains/<d>/tables/<t>.yaml      a table: columns, checks, rules, open questions, sample rows
enums/<e>.yaml                   allowed values, rendered as CHECK constraints
tools/                           model, schema, validate, ddl, migrate, explorer (+ explorer.template.html)
migrations/NNNN_<name>.sql       generated, committed with model.lock.json
model.lock.json                  the generated objects as of the last migration
build/                           schema.sql, explorer.html (generated, git-ignored)
tests/                           pytest; test_postgres.py needs a database
```

## Commands

Run from `datamodel/`, after `uv sync`. Without `make` (e.g. Windows), use
the right-hand column; it needs `PYTHONPATH=tools` (`$env:PYTHONPATH="tools"`
in PowerShell).

| make | Without make |
| --- | --- |
| `make validate` | `uv run python tools/validate.py` |
| `make ddl` | `uv run python tools/ddl.py` -> `build/schema.sql` |
| `make explorer` | `uv run python tools/explorer.py` -> `build/explorer.html` |
| `make migrate NAME=x` | `uv run python tools/migrate.py --name x` |
| `make migrate NAME=x ARGS=--allow-destructive` | `... --name x --allow-destructive` |
| `make check` | `uv run python tools/migrate.py --dry-run` must print `no changes` |
| `make test` | `uv run pytest -q -rs` |
| `make all` | validate, ddl, explorer |

`make check test` is the CI gate (`.github/workflows/datamodel.yml`).

### Postgres tests

`tests/test_postgres.py` checks the immutability triggers, and that the
baseline plus incremental migrations match a fresh build (`pg_dump -s`).
It runs when `DATAMODEL_TEST_DATABASE_URL` points at a maintenance database
on Postgres 16 (e.g. `postgresql://postgres:pw@localhost:5432/postgres`) and
`psql` and `pg_dump` are on PATH; otherwise it is skipped. Each test creates
and drops its own temporary databases.

## Changing the model

1. Edit the YAML.
2. `make validate`: errors name the table and column.
3. `make migrate NAME=<what_changed>`: writes the next
   `migrations/NNNN_<name>.sql` and updates `model.lock.json`. Destructive
   or risky changes (drops, type changes, `NOT NULL` without a default)
   are refused (exit 2) until reviewed and re-run with
   `ARGS=--allow-destructive`, which records the approval in the file.
4. Commit the YAML, the migration and `model.lock.json` together. CI fails
   if the YAML moved without a migration.

## Explorer

`make explorer` writes one self-contained `build/explorer.html`. Open it
straight from disk: it needs no server or network, and it isn't hosted or
uploaded anywhere.

## Rules

- **Synthetic data only.** Sample rows are examples, never real people.
  The validator rejects SSN-shaped values and emails outside reserved
  domains (`.invalid`, `example.com`, `example.org`).
- **Clinical scope is the CMO's.** A table can't be `published` while it
  has an open question owned by the CMO.
- Identifiers longer than 63 bytes are rejected; Postgres would silently
  truncate them.
