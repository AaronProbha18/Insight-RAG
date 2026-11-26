Issue: #21

## Why

Data models are maintained by hand, so DDL, migrations and review docs
drift apart, reviewers have no single diff for a model change, and nothing
stops an undocumented column existing in Postgres.

## What Changes

- New self-contained `datamodel/` folder, ported from the owner-supplied
  `modelkit` prototype: YAML model (domains, tables, enums) as the only
  source of truth, with tools that validate it and generate
  `build/schema.sql`, ordered `migrations/NNNN_*.sql` plus
  `model.lock.json`, and a static `build/explorer.html`.
- Own uv project (`datamodel/pyproject.toml`: pydantic, pyyaml; dev
  pytest) and a `Makefile` whose targets call `uv run`
  (`all validate ddl explorer migrate check test clean`).
- Gaps closed against the issue's acceptance criteria:
  - table and column names checked against the 63-byte limit (the
    prototype checks only generated constraint names);
  - Postgres 16 tests: immutability triggers, and baseline + incremental
    migrations equal a fresh build under `pg_dump -s`;
  - explorer escaping audited and checked by hand (opened locally).
- New CI workflow `.github/workflows/datamodel.yml` (paths `datamodel/**`):
  Postgres 16 service, `make check test`.
- ADR 0004 records YAML-as-source, CHECK-constraint enums, snapshot-diff
  migrations, the uuid key for this model, and the vanilla-JS explorer.
- Synthetic sample model (organization, person, membership; 6 tables,
  5 enums) ported as-is.

## Non-Goals

- Rename detection, data migrations, down migrations.
- A seed loader for `sample_rows`.
- ER diagrams; a browser review loop.
- Hosting the explorer, or uploading it from CI: it is a static page
  built and opened locally (Q2 settled by the product owner).
- Applying this model to Plumbline's own database or replacing its Alembic
  migrations.
- Any clinical decision; clinical scope is the CMO's.

## Capabilities

### New

- `datamodel-system`: YAML data model, its validator, and the DDL,
  migration and explorer generators.

### Modified

- none

## Risk

| Area | Yes/No | Note |
| --- | --- | --- |
| Answer quality (retrieval, ranking, chunking, embeddings, prompts, graph) | No | Separate folder; nothing in Plumbline's answering path changes. |
| Connector / ingestion | No | |
| Auth / document ACLs | No | The explorer describes schema, not data, and is only opened locally. |
| Public API or stream contract | No | No HTTP API. The `make` targets and YAML format are a new internal contract, recorded in ADR 0004. |
| Persistence / migration | Yes | Generates DDL and migrations for a Postgres schema (not Plumbline's); destructive changes need `--allow-destructive`. |
| New external dependency or service | Yes | pydantic, pyyaml, pytest in `datamodel/`; a Postgres 16 CI service. |

## Impact

New `datamodel/`, `.github/workflows/datamodel.yml`,
`docs/adr/0004-yaml-data-model.md`, `docs/architecture/overview.md`,
`README.md`, `AGENTS.md` (commands). Plumbline backend and frontend are
untouched. Architecture sign-off: business tech review.
