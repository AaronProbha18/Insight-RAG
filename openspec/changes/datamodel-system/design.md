## Context

The owner supplied a working prototype, `modelkit` (not in any repo):
`tools/{model,schema,validate,ddl,migrate,explorer}.py`,
`explorer.template.html`, 7 pytest tests, a Makefile, a CI workflow, a
synthetic sample model (3 domains, 6 tables, 5 enums), `0001_baseline.sql`
and `model.lock.json`. Read in full during triage. Compared with the
issue's acceptance criteria it lacks: identifier-length checks on table
and column names (only generated constraint/index/trigger names are
checked), any test against a real Postgres, and any explorer render test.
Plumbline's repo uses uv, ruff, and a `ci.yml` with a pgvector Postgres;
`main` requires green CI. Decisions that outlive this change:
[ADR 0004](../../../docs/adr/0004-yaml-data-model.md).

## Decisions

### Self-contained `datamodel/`

- **Choice**: everything under `datamodel/` with its own uv project and
  CI workflow; nothing imports from or into `backend/`.
- **Alternatives**: separate repo; inside `backend/`.
- **Why**: product owner chose this repo; isolation keeps Plumbline's
  database and Alembic untouched.
- **ADR**: docs/adr/0004-yaml-data-model.md

### Port verbatim first, then change

- **Choice**: task 1.1 commits the prototype unchanged (plus a uv
  project); later tasks change it, so each diff shows only the port
  changes. Tools keep importing siblings via `PYTHONPATH=tools` (set by
  the Makefile and in `pyproject` pytest config). No ruff pass: repo CI
  runs ruff only in `backend/`.
- **Alternatives**: restructure into a package or lint-clean while porting.
- **Why**: reviewable diffs; the prototype is small and works.
- **ADR**: none

### make calling uv

- **Choice**: `Makefile` targets run `uv run python tools/<x>.py`
  (`PY ?= uv run python`). README lists the same `uv run` commands for
  Windows without make.
- **Alternatives**: uv console scripts only.
- **Why**: the issue's acceptance criteria name `make` targets.
- **ADR**: docs/adr/0004-yaml-data-model.md

### Identifier length covers every name

- **Choice**: `validate.py` also checks table and column names (system
  columns included) against `PG_IDENT_MAX = 63` bytes, naming the table
  and column.
- **Why**: AC "identifier longer than 63 bytes fails (no silent
  truncation)".
- **ADR**: none

### Postgres 16 tests

- **Choice**: `tests/test_postgres.py`, marked `pg`, runs when
  `DATAMODEL_TEST_DATABASE_URL` is set (skipped otherwise). It creates
  temporary databases, then: (a) applies `migrations/*.sql` in order and
  checks immutability triggers (DELETE rejected; UPDATE outside
  `mutable_columns` rejected; allow-listed UPDATE succeeds and advances
  `updated_at`); (b) generates an incremental migration from a modified
  copy of the model, applies baseline + incremental to one database and a
  fresh `schema.sql` of the modified model to another, and compares
  `pg_dump -s` output with column order normalised.
- **Alternatives**: reuse the backend CI database.
- **Why**: product owner choice; keeps datamodel's Postgres version (16)
  independent of Plumbline's.
- UNVERIFIED - confirm the CI runner's `pg_dump` matches server 16
  (install `postgresql-client-16` if not) and how to normalise column
  order (sort column lines within each `CREATE TABLE`).

### Explorer: audit, then check by hand

- **Choice**: audit `explorer.template.html` so every interpolation of
  YAML text goes through `esc()` (JSON embedding already escapes `</` and
  `<!--`); then open `build/explorer.html` locally with a description
  containing `<script>` and `</script>` and record the result in
  `verification.md`.
- **Alternatives**: a Playwright test in CI.
- **Why**: product owner cut it - Chromium in CI for one check on a local
  review page isn't worth it. "No console errors" is a manual check.
- **ADR**: none
- UNVERIFIED - the sidebar render (`#side`) was not checked for `esc()`.

### Explorer is local only

- **Choice**: `make explorer` writes `build/explorer.html` (git-ignored);
  people open it from disk. No hosting and no CI artifact.
- **Alternatives**: CI artifact; static hosting behind Auth0.
- **Why**: product owner settled Q2: a static page viewed locally.
- **ADR**: docs/adr/0004-yaml-data-model.md

### CI workflow

- **Choice**: `.github/workflows/datamodel.yml`, on pull requests
  touching `datamodel/**` or the workflow: setup uv, Postgres 16 service,
  `make check test`.
- **Why**: AC "CI runs `make check test`"; the explorer is not uploaded
  (local only).
- UNVERIFIED - whether `.github/rulesets/main.json` must list the new
  job as a required check; a path-filtered required check blocks PRs that
  don't trigger it. Decide in task 4.1.

## Interfaces

```text
datamodel/
  Makefile                 all validate ddl explorer migrate check test clean
  pyproject.toml           uv project; pytest config (pythonpath=tools, marker pg)
  domains/<d>/domain.yaml
  domains/<d>/tables/<t>.yaml
  enums/<e>.yaml
  tools/{model,schema,validate,ddl,migrate,explorer}.py, explorer.template.html
  migrations/NNNN_<name>.sql + model.lock.json   (committed together)
  build/                   schema.sql, explorer.html (git-ignored)
  tests/
```

```bash
make validate                         # exit 0 / 1, errors name table.column
make migrate NAME=x [ARGS=--allow-destructive]   # exit 2 when refused
make check test                       # CI gate
```

## Data

The sample model's baseline (`migrations/0001_baseline.sql`) and
`model.lock.json` are ported unchanged. No effect on Plumbline's database.

## Risks / Open Questions

- Architecture/technical sign-off belongs to business tech review; the
  PR must not merge without it.
- The design replicates the CoreAevo Model Explorer from a partial read;
  formats and generator internals are new design and are UNVERIFIED
  against the real repo.
- Removing an enum value is not flagged as destructive (ADR 0004
  consequences); out of scope.
- Circular FKs between two tables rely on adding all FKs last.
