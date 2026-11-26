# datamodel

A YAML data model and the tools that check it and generate from it: the
DDL, ordered migrations, and a static explorer page. The YAML is the only
source; everything else is generated. Why it works this way:
[ADR 0004](../docs/adr/0004-yaml-data-model.md).

Plumbline's own tables stay in Alembic (`backend/migrations/`), which is
their source. `models/plumbline/` is a YAML **mirror** of that schema, and
CI fails when the two disagree (see [the drift rule](#plumblines-model-and-the-drift-rule)).
This folder is a separate uv project, and nothing here imports from `backend/`.

## Layout

```text
models/<m>/model.yaml                     optional per-model settings (see Model settings)
models/<m>/domains/<d>/domain.yaml        a domain: layer, load order, depends_on
models/<m>/domains/<d>/tables/<t>.yaml    a table: columns, checks, rules, open questions, sample rows
models/<m>/enums/<e>.yaml                 allowed values, rendered as CHECK constraints
models/<m>/migrations/NNNN_<name>.sql     generated, committed with model.lock.json (sample only)
models/<m>/model.lock.json                the generated objects as of the last migration (sample only)
models/<m>/build/                         schema.sql, explorer.html (generated, git-ignored)
tools/                                    model, schema, validate, ddl, migrate, explorer, drift (+ explorer.template.html)
tests/                                    pytest; test_postgres.py and test_drift.py need a database
```

## Commands

Run from `datamodel/`, after `uv sync`. Every target works on one model:
`MODEL=models/plumbline` (default `models/sample`), or `--model <dir>` on the
tools. Without `make` (e.g. Windows), use the right-hand column; it needs
`PYTHONPATH=tools` (`$env:PYTHONPATH="tools"` in PowerShell).

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
| (no target) | `uv run python tools/drift.py <url-a> <url-b>`: exit 0 same structure, 1 drift, 2 `psql` failed |

`make check test` is the CI gate (`.github/workflows/datamodel.yml`).

### Postgres tests

`tests/test_postgres.py` checks the immutability triggers, and that the
baseline plus incremental migrations match a fresh build (`pg_dump -s`);
`tests/test_drift.py` runs `tools/drift.py` on pairs of databases.
They run when `DATAMODEL_TEST_DATABASE_URL` points at a maintenance database
on Postgres 16 (e.g. `postgresql://postgres:pw@localhost:5432/postgres`) and
`psql` and `pg_dump` are on PATH; otherwise it is skipped. Each test creates
and drops its own temporary databases.

## Model settings

An optional `models/<m>/model.yaml` sets conventions. Every key is optional;
the defaults are the sample model's behaviour, and an unknown key is an error.

| Key | Values (default first) | Effect |
| --- | --- | --- |
| `id_type` | `uuid`, `bigserial` | the injected `id`; foreign keys must then be `uuid` or `bigint` |
| `public_id_check` | `true`, `false` | the `ck_<t>_public_id` CHECK |
| `updated_at_trigger` | `true`, `false` | `tg_<t>_updated_at` and `set_updated_at()` |
| `comments` | `true`, `false` | `COMMENT ON` every table and column |
| `constraint_names` | `generated`, `postgres` | `pk_`/`fk_`/`uq_`/`ck_` names, or `<t>_pkey`, `<t>_<c>_fkey`, `<t>_<cols>_key`, `<t>_<c>_check` |

A table can set `system_columns` to a subset of `public_id`, `created_at`
and `updated_at` (default: all three; `id` is always injected). `public_id_prefix`
is required only when `public_id` is injected.

Column types beyond the basics: `char(n)`, `double precision`, `vector(n)`
and `vector({embedding_dim})`. `{embedding_dim}` is filled from
`PLUMBLINE_EMBEDDING_DIM`; `ddl` and `explorer` fail naming that variable if
it's unset or isn't a positive integer, while `validate` doesn't need it.

## Plumbline's model and the drift rule

`models/plumbline/` mirrors the Alembic schema in `backend/migrations/`.
**Change Alembic and the YAML in the same PR.** The `plumbline-drift` job in
`.github/workflows/datamodel.yml` fails otherwise. It applies Alembic head to
one database, loads the DDL generated from `models/plumbline` into another
(`PLUMBLINE_EMBEDDING_DIM=8`, `CREATE EXTENSION vector` first), and runs
`tools/drift.py` on the two. Columns (type, nullability, default),
constraint definitions, indexes and triggers are compared; object names,
comments, column order and `alembic_version` are ignored. Trigger function
bodies, sequences and extensions are not compared. The job is not a required
check, so look for it green on the PR.

`models/plumbline` has no `migrations/` or `model.lock.json`: Alembic owns
the migrations, so `make migrate` and `make check` apply to `models/sample`
only. To run the same comparison locally, point `drift.py` at two Postgres
databases with pgvector, one built by `alembic upgrade head` and one from
`PLUMBLINE_EMBEDDING_DIM=8 make ddl MODEL=models/plumbline`.

## Changing the model

1. Edit the YAML.
2. `make validate`: errors name the table and column.
3. `make migrate NAME=<what_changed>` (sample model): writes the next
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
