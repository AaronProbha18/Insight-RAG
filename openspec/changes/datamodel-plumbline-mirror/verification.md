## Checks Run

All run in this session on branch `feat/28-datamodel-plumbline-mirror`, after task 6.6. `make` is not installed on this machine, so each Makefile target was run as the command it wraps (the Makefile passes `--model $(MODEL)` and nothing else).

| Command | Result |
| --- | --- |
| `cd datamodel && PYTHONPATH=tools uv run python -m pytest -q -rs tests` (no database) | 45 passed, 4 skipped (the Postgres tests: `DATAMODEL_TEST_DATABASE_URL` not set) |
| Same suite inside a throwaway `pgvector/pgvector:pg18` container, `DATAMODEL_TEST_DATABASE_URL` set, `psql` present | 49 passed, 0 skipped |
| `tools/validate.py` (no `--model`, so `models/plumbline`) | 1 domains, 4 tables, 0 errors |
| `PLUMBLINE_EMBEDDING_DIM=8 tools/ddl.py` (no `--model`) | wrote `models/plumbline/build/schema.sql` (4 tables) |
| `tools/ddl.py` with `PLUMBLINE_EMBEDDING_DIM` unset | `ValueError: embedding: PLUMBLINE_EMBEDDING_DIM must be set to a positive integer; got ''` |
| `PLUMBLINE_EMBEDDING_DIM=768 tools/explorer.py` | wrote `models/plumbline/build/explorer.html` (21 KB) |
| Explorer rendered in Node with a stub DOM: 6 views (overview, open questions, 4 tables) x 3 tabs | 18 views rendered, 0 containing `undefined`/`NaN` |
| Drift job steps in the container: `alembic upgrade head` into DB `alembic` (2 revisions run), generated DDL into DB `generated` after `CREATE EXTENSION vector`, then `tools/drift.py <alembic> <generated> --label-a alembic --label-b yaml` (the CI command) | `no drift`, exit 0 |
| Same, in a scratch copy only, with `feedback.rating` made nullable | `table feedback: column rating differs: alembic=['text', True, None] yaml=['text', False, None]`, exit 1 |
| `openspec validate datamodel-plumbline-mirror` | valid |
| `openspec validate --all` | 11 passed, 0 failed |
| `python scripts/check_links.py` | `links ok (117 files)` |
| `cd backend && uv run python -m ruff check .` / `ruff format --check .` | All checks passed; 61 files already formatted |
| `cd backend && uv run python -m pytest -q -rs` (dev db up via `docker compose up -d db`) | 199 passed, 1 skipped (needs Ollama) |
| `cd frontend && npm run lint` / `npm run build` / `npm test` | lint clean; build succeeded; 18 test files, 71 tests passed |
| `git diff --name-only main -- backend frontend` | 0 files: this change touches neither |
| `git diff --name-only origin/main...HEAD` outside `datamodel/`, `docs/`, `openspec/`, `.github/`, `AGENTS.md` | none (the skills commit is not on this branch) |

Not run, so not claimed:
- `make validate test` / `make check` (no `make` here; CI runs `make validate test`).
- The GitHub Actions `plumbline-drift` job and the `datamodel` job. The loop doesn't push, and nothing has been pushed since `8e748d3`: the commits after it are local only.
- The Postgres-backed tests on Postgres 16 (the `datamodel` CI job's version). Locally they ran on 18 with pgvector.
- The explorer in a real browser; the Node render check is a stub DOM, not a browser.

## Scenario Coverage

Scenarios of `specs/datamodel-plumbline-mirror/spec.md`:

| Spec scenario | Test |
| --- | --- |
| Model selection: Default is Plumbline's model | `test_generate.py::test_default_model_is_plumbline` |
| Model selection: Named model | `test_generate.py::test_ddl_model_flag_writes_into_model_build` |
| Plumbline conventions: Plumbline conventions generated | `test_conventions.py::test_plumbline_conventions_generated` |
| Plumbline conventions: uuid foreign key rejected | `test_conventions.py::test_non_bigint_foreign_key_rejected` (an `int` foreign key fails naming `note.account_id` and `bigint`); `test_toolkit_scope.py::test_uuid_column_rejected`. **Partial:** a literal `uuid` foreign key now fails earlier, at load, as an unsupported type, not with the `bigint` message. |
| Per-table system columns: Subset injected | `test_conventions.py::test_system_columns_subset` |
| Per-table system columns: Unknown entry | `test_conventions.py::test_system_columns_bad_entry[id]` and `[bogus]` |
| Extra column types: Embedding dimension filled | `test_conventions.py::test_embedding_dim_filled` |
| Extra column types: Dimension missing | `test_conventions.py::test_embedding_dim_missing[None]` and `[0]`; `test_validate_needs_no_embedding_dim` |
| Plumbline model mirrors Alembic: Model valid | `test_plumbline_model.py::test_plumbline_model`; `validate` exits 0 (above) |
| Drift gate: In sync | container run above (`no drift`, exit 0); the GitHub run is pending |
| Drift gate: Alembic changed without YAML | `test_drift.py::test_extra_column_is_drift`, `test_extra_column_named`. **Mechanism only:** tested on two plain databases; no test adds a column to a real Alembic revision. |
| Drift gate: YAML changed without Alembic | `test_drift.py::test_nullability_change_is_drift`, and the negative run above on the real Alembic and generated pair (exit 1, naming `feedback` and `rating`) |
| Drift gate: Names and comments ignored | `test_drift.py::test_names_and_comments_are_not_drift` |
| Drift gate: Output names the sides | `test_drift.py::test_labels_name_the_sides`, `test_cli_labels`, and the negative run above (`alembic=...` and `yaml=...`) |
| Toolkit limited: Removed field rejected | `test_toolkit_scope.py::test_removed_table_field_rejected[immutable|mutable_columns|business_rules|sample_rows]`, `test_removed_column_field_rejected` |
| Toolkit limited: Migration tooling absent | `test_toolkit_scope.py::test_migration_tooling_absent` (plus `test_model_yaml_rejected` for the settings file) |

Scenarios of `specs/datamodel-system/spec.md` (ADDED "Model validation" and "Generated DDL"; the REMOVED requirements have no scenarios):

| Spec scenario | Test |
| --- | --- |
| Model validation: Valid model | `test_validate.py::test_valid_model`, `test_model.py::test_mini_fixture_is_valid` |
| Model validation: Foreign key outside depends_on, not at id, or not bigint | `test_validate.py::test_rule_fails[fk outside depends_on]`, `[fk not at id]`; `test_conventions.py::test_non_bigint_foreign_key_rejected` |
| Model validation: Identifier too long | `test_validate.py::test_rule_fails[table name 64 bytes]`, `[column name 64 bytes]`, the two 63-byte cases; `test_model.py::test_long_constraint_name_rejected` |
| Model validation: Duplicate prefix or question id | `test_validate.py::test_rule_fails[duplicate prefix]`. **Gap: no test for a duplicate `OQ-<TABLE>-` id.** The code (`validate.py`) still checks it; the only test of that shared check was the duplicate-rule-id case, deleted in 6.1 with `business_rules`. |
| Model validation: Default on a direct identifier or PHI column | `test_validate.py::test_rule_fails[default on pii column]`, `test_model.py::test_pii_default_rejected` |
| Model validation: Published with an open CMO question | `test_validate.py::test_rule_fails[published with open CMO question]` |
| Model validation: Answered question without an answer | `test_validate.py::test_rule_fails[answered without answer]` |
| Generated DDL: System columns injected | `test_generate.py::test_system_columns_injected`, `test_conventions.py::test_system_columns_subset` |
| Generated DDL: Postgres-style names and no extras | `test_conventions.py::test_plumbline_conventions_generated` |

The unchanged main-spec requirement **Explorer** ("Offline render", no console errors, YAML escaped) has no automated test; its coverage this session is the Node render check above (stub DOM, not a browser).

## Eval Results

N/A per eval-plan: schema tooling and a CI gate; nothing in Plumbline's retrieval, prompts or graph changes.

## Deviations

- **Scope grew during implementation (owner decisions), tasks 6.1 to 6.6.** After the mirror, drift gate and conventions settings (tasks 1 to 5) were built, the owner chose to remove everything Plumbline doesn't use. Removed: `models/sample/`, `migrate.py`, `model.lock.json` and migrations, enums, immutable tables, uuid ids, `sample_rows`, `business_rules`, and the `model.yaml` settings (so task 2.1 and parts of 1.1 and 3.1 are superseded). `proposal.md`, `design.md`, the spec and `tasks.md` were revised first and say so.
- **Beyond the task text:** the column type list was slimmed to what Plumbline uses (`uuid`, `date`, `time`, `inet`, `numeric`, `varchar` gone); the generated DDL lost its `CREATE EXTENSION IF NOT EXISTS pgcrypto;` line (Plumbline's schema never used it); a `model.yaml` in a model folder now makes `load()` raise; `make check` and the CI wording moved from 6.2 into 6.1.
- **`indexes` support was kept** although Plumbline has no indexes today (the overview names an HNSW index as a later migration, and the drift gate compares indexes).
- **The `datamodel-system` delta uses REMOVED plus ADDED requirements, not MODIFIED**, because OpenSpec refuses a MODIFIED block that drops scenarios.
- **ADR 0004 was rewritten in place and accepted in this PR** (with ADRs 0002 and 0003), instead of a new ADR 0005. The revision note says it was revised before any release. Acceptance is the owner's request; the business tech review's sign-off is this PR's approval.
- **Alembic-comparison test dropped (3.1):** `datamodel/` never reads `backend/`, so the drift gate is the comparison.
- **The drift gate ignores column order** and `public_id_prefix` is optional (required only with `public_id`), both decided during implementation.
- **`answers` is `status: draft`** with open question `OQ-ANSWERS-001` (owner CMO) about whether the free-text columns are PII `indirect`. It is a default taken when approving the plan, not a decision.
- **The skills change (`db-change` rules) was moved out of this PR** into a separate chore (local branch `skills-db-change-wip`); this branch contains no `.claude/` changes.
- **Pending, needs a push (the loop never pushes):** the green `plumbline-drift` and `datamodel` jobs on the PR, and the scratch commit that makes a YAML column nullable and fails the drift job (then reverted).
- **Gaps to decide:** no test for a duplicate `OQ-` id, none for the explorer outside the stub-DOM check, and none that adds a column to a real Alembic revision. Each is one small test through rag-plan if wanted before the PR.

## Verdict

PASS, with the gaps and pending CI evidence listed above. Every check I ran passed, and every spec scenario has at least one test or a recorded real run. Three are partial: "Duplicate prefix or question id" (no question-id test), "uuid foreign key rejected" (a literal `uuid` fails at load, so the `bigint` message is shown for an `int` key), and the "Alembic changed without YAML" mechanism test. If you want the duplicate question id covered strictly, add one test before the PR.
