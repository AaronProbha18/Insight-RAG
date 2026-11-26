## Task
5.1 Extend ADR 0004 (still Proposed, business tech review) with the mirror decision, `models/` layout and conventions; `datamodel/README.md` (models, `MODEL=`, `model.yaml`, the drift rule: "change Alembic and the YAML in the same PR"); overview and `AGENTS.md` (backend schema changes now need the YAML too); backend development guide's migration section. Test: `python scripts/check_links.py` passes.

## Scope
Docs only. Every statement below describes code that already landed in tasks 1.1 to 4.2.

- **`docs/adr/0004-yaml-data-model.md`, extended in place** (still Proposed; folded in because 0004 isn't Accepted, so AGENTS.md rule 5 allows correcting it, and one ADR is one sign-off by business tech review):
  - Status: `Proposed. Introduced by the datamodel-system change (issue #21), extended by datamodel-plumbline-mirror (issue #28). Architecture and technical sign-off: business tech review.`
  - Context: `datamodel/` is no longer separate from Plumbline's schema; it also holds a mirror of it. The line "It lives in datamodel/, separate from Plumbline's own database, which keeps its Alembic migrations" is reworded (it still doesn't touch Plumbline's database).
  - Decision, reworded: "YAML is the model" paths become `datamodel/models/<name>/domains/**` and `.../enums/*`. The sentence "A column in the database but not in a table.yaml is a defect" is scoped to models whose YAML is the source (`models/sample`); for `models/plumbline` Alembic is the source. The System columns bullet says the injected columns follow `model.yaml` and per-table `system_columns` (the defaults are the ones listed), replacing "Plumbline's own tables use bigserial; this ADR governs datamodel/ only".
  - Decision, added bullets: models live under `datamodel/models/<name>/` with `--model`/`MODEL=`; conventions are `model.yaml` settings (`id_type`, `public_id_check`, `updated_at_trigger`, `constraint_names`, `comments`); new types `char(n)`, `double precision`, `vector(n)`, `vector({embedding_dim})` filled from `PLUMBLINE_EMBEDDING_DIM`; **Alembic stays the source for Plumbline's schema, `models/plumbline/` mirrors it, and the `plumbline-drift` CI job fails on disagreement** (columns, types, nullability, defaults, constraint definitions, indexes, triggers; names, comments and column order ignored); `datamodel/` still never reads `backend/`.
  - Alternatives, added: YAML as the source with generated migrations (a cutover on a live schema; `vector(N)` has no clean home) and docs only with no gate (drifts).
  - Consequences, added: every Plumbline schema change is written twice, and forgetting the YAML fails CI; the gate isn't a required check, so a reviewer must see it green; it doesn't compare trigger function bodies, sequences or extensions.
- **`datamodel/README.md`:**
  - Intro and the "not Plumbline's schema" paragraph: it now also contains a model that mirrors Plumbline's Alembic schema; Alembic stays the source; `nothing here imports from backend/` still holds.
  - Layout: `models/<name>/` with `domains/`, `enums/`, optional `model.yaml`, `migrations/`, `model.lock.json`, `build/`; `models/sample` and `models/plumbline`; `tools/` adds `drift`.
  - Commands: a `MODEL=` note (`make validate|ddl|explorer|migrate|check MODEL=models/plumbline`, default `models/sample`) and a `drift` row; the without-make column gets `--model`.
  - New "Model settings" section: the five `model.yaml` keys with defaults, `system_columns`, the new types, and that `PLUMBLINE_EMBEDDING_DIM` is needed to generate DDL or the explorer for a model with `vector({embedding_dim})` (not to validate).
  - New "Plumbline's model and the drift rule" section: change Alembic and the YAML in the same PR; the `plumbline-drift` job; how to run it locally (the steps from the job); `models/plumbline` has no `migrations/` or `model.lock.json`.
  - "Postgres tests": add `test_drift.py`; note the Postgres 16 mention is the CI job's, drift job uses pgvector `pg18`.
- **`docs/architecture/overview.md`** ("Data model and migrations", lines 177 to 182): replace the "not Plumbline's schema ... Whether to align the two is #28" paragraph. It now says `datamodel/models/plumbline/` mirrors Alembic, the `plumbline-drift` check keeps the two in step, and links ADR 0004 and the README. The `datamodel/` toolkit still doesn't touch Plumbline's database at runtime.
- **`AGENTS.md`:** the `datamodel/` layout line (50) updated; a new bullet under workflow rule 7 or beside rule 1... exact placement: one added sentence in rule 7, "A backend schema change (a new Alembic revision) also updates `datamodel/models/plumbline/` in the same PR; `plumbline-drift` fails otherwise." Also the datamodel commands block (153 to 157) mentions `MODEL=`.
- **`docs/guides/backend-development-guide.md`** section 7: a new rule **7.5**: "A new Alembic revision also updates `datamodel/models/plumbline/` in the same PR; the `plumbline-drift` CI job fails when they disagree (ADR 0004)." The Do/Don't table gets one row: "Update `datamodel/models/plumbline/` with the revision" / "Merge a revision the YAML doesn't mirror".
- Guides read: backend development guide (section 7 and its table), `docs/architecture/overview.md`, `AGENTS.md` rules 5 and 7. This is a docs task; there's no code to follow a guide for.

## Not in scope
- Accepting ADR 0004 (status stays Proposed; business tech review decides).
- Code changes, `README.md` at the repo root (it only links `datamodel/README.md`, which stays accurate), and the `openspec/specs` sync (that happens at archive, per the repo's "archive before merge" rule).
- Anything about CI settings (the drift job is not a required check; ADR 0004 states it).

## Files
| File | Change |
| --- | --- |
| `docs/adr/0004-yaml-data-model.md` | extended with the mirror decision (status, context, decisions, alternatives, consequences) |
| `datamodel/README.md` | models, `MODEL=`, settings, drift rule |
| `docs/architecture/overview.md` | the `datamodel/` paragraph |
| `AGENTS.md` | layout line; rule 7 sentence; commands block |
| `docs/guides/backend-development-guide.md` | rule 7.5 and one table row |

## Tests
- `python scripts/check_links.py` passes (the task's test), run from the repo root.
- Facts check: every command, path, setting and default written in the docs matches the code. I'll grep each (e.g. `Settings` in `schema.py`, the Makefile targets, the job name in `datamodel.yml`) before finishing, and report any doc claim I couldn't confirm.
- `uv run python -m pytest -q -rs tests` in `datamodel/` still passes (nothing in code changed, so this confirms I didn't touch it).

## Checks to run
- `python scripts/check_links.py`
- `cd datamodel && uv run python -m pytest -q -rs tests`
- `git diff --stat` shows docs files only.

## Risks / Questions
- Folding means a reviewer who already read 0004 sees its decisions change. It's Proposed, git keeps the history, and the Status line names both changes. The change's `proposal.md`, `design.md` and `tasks.md` 5.1 now point at 0004 instead of a new 0005.
- The local-run recipe for the drift job uses Docker and a copied-in Python on my machine, which I won't document. The README will describe the job's own steps against any Postgres with pgvector.
