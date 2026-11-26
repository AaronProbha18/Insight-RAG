## Task
6.6 Docs: rewrite ADR 0004 in place with a revision note (Alembic is the source, the YAML mirrors it, the toolkit is limited to what that needs); update `datamodel/README.md`, `AGENTS.md` and `docs/architecture/overview.md`; create the delta spec `specs/datamodel-system/spec.md` in this change (REMOVED: Migrations; MODIFIED: validator, DDL generation and explorer requirements, plus CI gate to match). Test: `python scripts/check_links.py` and `openspec validate datamodel-plumbline-mirror` pass.

## Scope
Docs, one explorer sentence, and the delta spec. Every statement describes code that landed in 6.1 to 6.5.

- **ADR 0004** (`docs/adr/0004-yaml-data-model.md`), rewritten in place; Status stays Accepted, with a revision note: "Revised in the same PR that accepted it (issue #28, before release): the toolkit was trimmed to what Plumbline's mirror needs, and the sample model, migration generator, enums, immutable tables, uuid ids, sample rows, business rules and per-model settings were removed."
  - **Context:** `datamodel/` holds a YAML mirror of Plumbline's Alembic schema and the tools that check it. It never touches Plumbline's database outside the CI job and never reads `backend/`.
  - **Decision bullets:**
    - Alembic stays the source of Plumbline's schema; `models/plumbline/` mirrors it; the `plumbline-drift` job fails on disagreement (what it compares and ignores, with the `alembic`/`yaml` labels). A schema change updates Alembic and the YAML in the same PR.
    - Models live under `datamodel/models/<name>/` (`domains/` and a git-ignored `build/`); tools take `--model`, default `models/plumbline`.
    - System columns: `id bigserial` always; `public_id`, `created_at`, `updated_at` per table's `system_columns`; foreign keys are `bigint` and point at `id`.
    - One set of conventions: no `public_id` CHECK, no `updated_at` trigger, no comments, Postgres-style constraint names.
    - Types: `text`, `int`, `bigint`, `bool`, `timestamptz`, `jsonb`, `char(n)`, `double precision`, `vector(n)`, `vector({embedding_dim})` (from `PLUMBLINE_EMBEDDING_DIM`).
    - The explorer is one static HTML file, local only (kept as is).
    - Commands are `make` targets calling `uv run`: `validate`, `ddl`, `explorer`, `test`.
  - **Alternatives:** the earlier list (SQL first, modelling tool, Postgres enums, Alembic autogenerate or Atlas, React explorer) is trimmed to what still applies; keep "YAML as the source with generated migrations" and "docs only with no gate"; add "keep the generator, sample and per-model settings" (rejected: code and a model Plumbline never uses).
  - **Consequences:** every Plumbline schema change is written twice; the drift job isn't a required check; it doesn't compare functions, sequences or extensions; the toolkit is smaller than #21 first delivered, and a feature that comes back needs a reason.
- **`datamodel/README.md`**, rewritten: purpose (a mirror and its checks, not a generator of Plumbline's schema); layout (`models/plumbline/`, `tests/fixtures/mini/`, `tools/`); commands (`validate`, `ddl`, `explorer`, `test`, `all`, plus `drift.py` with the label flags); Postgres tests (what `test_postgres.py` and `test_drift.py` do, Postgres needs, CI uses 16 for the tests and 18 with pgvector for the drift job); what a table YAML contains (`columns`, `checks`, `unique`, `indexes`, `system_columns`, `public_id_prefix`, `open_questions`, types, `{embedding_dim}`); the drift rule; changing the model (Alembic first, then the YAML in the same PR, then `make validate`, `make ddl` and the local drift check); the explorer; rules (clinical scope is the CMO's, identifiers over 63 bytes rejected). Gone: model settings, enums, migrations, `make check`, sample-row rules.
- **`AGENTS.md`:** the datamodel commands block (`make check test` becomes `make validate test`; `migrate` leaves the "without make" list; `MODEL=` default is `models/plumbline`). Layout line 50 and the rule 7 sentence are already right.
- **`docs/architecture/overview.md`:** the `datamodel/` paragraph says "generates DDL, migrations and an explorer"; it becomes "checks and generates DDL and an explorer".
- **Explorer template:** the overview callout "The YAML is the model ... A column that exists in the database but not in a table.yaml is a defect" becomes "This model mirrors Plumbline's Alembic schema. DDL and this page are generated from `domains/**/*.yaml`. A column in only one of the two is drift, and CI fails."
- **Delta spec** `specs/datamodel-system/spec.md` in this change:
  - `MODIFIED` **Validator gates the model:** the scenarios keep valid model, foreign key (outside `depends_on` or not at `id`, and not `bigint`), identifier too long, duplicate prefix or question id, default on a direct/PHI column, published with an open CMO question, answered question without an answer. Dropped: "Bad sample row" and the rule-id part of the duplicate scenario.
  - `MODIFIED` **DDL generation:** system columns injected per `system_columns` with `id bigserial`, Postgres-style names and no comments, public_id CHECK or updated_at trigger; the "Immutable table trigger" and "Enum as CHECK" scenarios are gone.
  - `REMOVED` **Migrations** and **CI gate** (each with Reason and Migration: the drift gate in `datamodel-plumbline-mirror` replaces the CI gate; Plumbline's migrations are Alembic's).
  - **Explorer:** unchanged, so not in the delta.
- Guides read: none apply to the doc edits; `docs/guides/backend-development-guide.md` rule 7.5 (added in 5.1) is still accurate, so it isn't touched. I read the current ADR 0004, the README, the `AGENTS.md` and overview passages, and the main `datamodel-system` spec.

## Not in scope
- Archiving the change and syncing the main spec (that happens after verify, on this branch, per the archive-before-merge rule).
- `verification.md` (rewritten by `rag-verify` at the end) and the PR body.
- Code beyond the one template sentence.

## Files
| File | Change |
| --- | --- |
| `docs/adr/0004-yaml-data-model.md` | rewritten in place, revision note |
| `datamodel/README.md` | rewritten |
| `AGENTS.md` | the datamodel commands block |
| `docs/architecture/overview.md` | one paragraph |
| `datamodel/tools/explorer.template.html` | one callout sentence |
| `openspec/changes/datamodel-plumbline-mirror/specs/datamodel-system/spec.md` | new delta spec |

## Tests
- `python scripts/check_links.py` passes.
- `openspec validate datamodel-plumbline-mirror` passes with the new delta spec (this also checks the MODIFIED and REMOVED requirement names match the main spec's exactly).
- Fact check, run and reported: every command, flag, path and file named in the README and ADR exists in `datamodel/` (the Makefile targets, `drift.py --label-a/--label-b`, `tests/fixtures/mini/`, the job name `plumbline-drift`, the Postgres versions in `datamodel.yml`). I'll grep each and list anything I couldn't confirm.
- Rebuild the explorer and re-run the Node render check (all pages render, no `undefined`), plus a grep for the new callout text.
- `cd datamodel && PYTHONPATH=tools uv run python -m pytest -q -rs tests` still passes (no code changed, so this only confirms the template edit broke nothing).

## Checks to run
- The four above, and `git diff --stat` showing only the files in the table.

## Risks / Questions
- **ADR 0004 is Accepted**, so this is a visible change to an Accepted decision. It's justified because the acceptance and the revision land in the same PR and nothing was released in between; the revision note says so. If the business tech review has already read the earlier text, they'll see a different ADR, so the PR body should say it.
- The `datamodel-system` delta leaves the main spec's Explorer requirement as is; its "no console errors" scenario was last checked by my Node render check, not a browser.
- `docs/architecture/overview.md` and the backend guide still describe Plumbline's Alembic migrations, which is right and unchanged.
