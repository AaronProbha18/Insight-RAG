## Checks Run

All run in this session on branch `feat/28-datamodel-plumbline-mirror`. `make` is not installed on this machine, so each Makefile target was run as the command it wraps (the Makefile passes `--model $(MODEL)` and nothing else).

| Command | Result |
| --- | --- |
| `cd datamodel && PYTHONPATH=tools uv run python -m pytest -q -rs tests` (no database) | 50 passed, 6 skipped (the Postgres tests: `DATAMODEL_TEST_DATABASE_URL` not set) |
| Same suite inside a throwaway `pgvector/pgvector:pg18` container, `DATAMODEL_TEST_DATABASE_URL` set, `psql` present | 56 passed, 0 skipped |
| `tools/validate.py` (sample) | 3 domains, 6 tables, 5 enums, 0 errors |
| `tools/migrate.py --dry-run` (sample; what `make check` runs) | `no changes` |
| `tools/validate.py --model models/plumbline` | 1 domains, 4 tables, 0 enums, 0 errors |
| `PLUMBLINE_EMBEDDING_DIM=8 tools/ddl.py --model models/plumbline` | wrote `models/plumbline/build/schema.sql` (4 tables) |
| `tools/ddl.py --model models/plumbline` with the variable unset | `ValueError: embedding: PLUMBLINE_EMBEDDING_DIM must be set to a positive integer; got ''` |
| `PLUMBLINE_EMBEDDING_DIM=8 tools/explorer.py --model models/plumbline` | wrote `models/plumbline/build/explorer.html` (25 KB) |
| Drift job steps in the container: `alembic upgrade head` into DB `alembic` (2 revisions run), generated DDL into DB `generated` after `CREATE EXTENSION vector`, `tools/drift.py <alembic> <generated>` | `no drift`, exit 0 |
| Same, in a scratch copy only, with `feedback.rating` made nullable | `table feedback: column rating differs: a=['text', True, None] b=['text', False, None]`, exit 1 |
| `openspec validate datamodel-plumbline-mirror` | valid |
| `openspec validate --all` | 11 passed, 0 failed |
| `python scripts/check_links.py` | `links ok (115 files)` |
| `cd backend && uv run ruff check .` / `uv run ruff format --check .` | All checks passed; 61 files already formatted |
| `cd backend && uv run python -m pytest -q -rs` (dev db started with `docker compose up -d db`) | 199 passed, 1 skipped (needs Ollama) |
| `cd frontend && npm run lint` / `npm run build` / `npm test` | lint clean; build succeeded; 18 test files, 71 tests passed |
| `git diff --name-only main -- backend frontend` | 0 files: this change touches neither |

Not run, so not claimed:
- `make check test` (no `make` here; CI runs it).
- The GitHub Actions `plumbline-drift` job. The loop doesn't push, so there's no CI run yet.
- The existing `datamodel` CI job on Postgres 16. Locally the Postgres tests ran on 18 with pgvector.
- An earlier backend `pytest` attempt (with `--active`) failed on a `uv` trampoline error, and the first plain run had 3 failures from an unreachable Postgres. After starting the dev database, the run above passed. Those two earlier runs are not counted as results.

## Scenario Coverage

| Spec scenario | Test |
| --- | --- |
| Model selection: Default model unchanged | the whole existing datamodel suite on `workcopy` of `models/sample`; `migrate --dry-run` prints `no changes` |
| Model selection: Named model | `test_generate.py::test_ddl_model_flag_writes_into_model_build` (`ddl.py --model` writes into that model's `build/`); `test_plumbline_model.py::test_plumbline_model` (validate and DDL of `models/plumbline`). **Partial: no automated test runs `explorer.py --model`.** It was run by hand (above), and no existing test calls the explorer at all. |
| Per-model conventions: No model.yaml keeps today's output | `test_conventions.py::test_no_model_yaml_keeps_lock` |
| Per-model conventions: bigserial keys | `test_conventions.py::test_bigserial_ids_and_bigint_fks` |
| Per-model conventions: Conventions switched off | `test_public_id_check_off`, `test_updated_at_trigger_off`, `test_comments_off`, `test_postgres_constraint_names` (one per setting); all five together in `models/plumbline`, exercised by the drift run |
| Per-table system columns: Subset injected | `test_conventions.py::test_system_columns_subset` |
| Per-table system columns: Unknown entry | `test_conventions.py::test_system_columns_bad_entry[id]` and `[bogus]` |
| Extra column types: Embedding dimension filled | `test_conventions.py::test_embedding_dim_filled` (`vector(8)`, `char(64)`, `double precision`) |
| Extra column types: Dimension missing | `test_conventions.py::test_embedding_dim_missing[None]` and `[0]`; `test_validate_needs_no_embedding_dim` |
| Plumbline model mirrors Alembic: Model valid | `test_plumbline_model.py::test_plumbline_model`; `validate --model models/plumbline` exits 0 |
| Drift gate: In sync | drift positive run above (`no drift`, exit 0); the CI run is pending |
| Drift gate: Alembic changed without YAML | `test_drift.py::test_extra_column_is_drift` (an extra column on one side exits 1, naming it) and `test_extra_column_named`. This tests the mechanism on two plain databases; no test adds a column to a real Alembic revision. |
| Drift gate: YAML changed without Alembic | `test_drift.py::test_nullability_change_is_drift`, and the negative run above on the real Alembic and generated pair (exit 1, naming `feedback` and `rating`) |
| Drift gate: Names and comments ignored | `test_drift.py::test_names_and_comments_are_not_drift` |

## Eval Results

N/A per eval-plan: schema tooling and a CI gate; nothing in Plumbline's retrieval, prompts or graph changes.

## Deviations

- **Alembic-comparison test dropped (task 3.1).** The task asked for a test comparing `models/plumbline` with the Alembic SQL, but design.md says `datamodel/` never reads `backend/`. The owner chose to rely on the drift gate, and `tasks.md` 3.1 was reworded. The gate has now run once for real (above).
- **No separate ADR 0005 (task 5.1).** design.md and the proposal named a new ADR 0005. The owner chose to fold the decision into ADR 0004, still Proposed; `proposal.md`, `design.md` and `tasks.md` were updated.
- **`public_id_prefix` is optional** (task 2.2, not in `tasks.md`): required only when `public_id` is injected. Plumbline's `chunks` and `feedback` have no `public_id`.
- **Drift gate ignores column order.** This settles the question left open in 3.1.
- **`feedback` checks written without `IS NULL OR`** (task 4.2): the first real comparison found Postgres renders the explicit form differently from Alembic's column checks, so the YAML now matches Alembic's expressions.
- **`answers` is `status: draft`** with open question `OQ-ANSWERS-001` (owner CMO): free-text columns may hold health details, and whether to class them as PII `indirect` is unanswered. `validate.py` forbids a published table with an open CMO question. This is a default taken when approving the plan, not a decision.
- **Drift tests use plain tables**, so they don't need pgvector; the gate itself runs on pgvector in CI.
- **Pending, needs a push (the loop never pushes):** the green `plumbline-drift` run on the PR, and the scratch commit that makes a YAML column nullable and fails the job (then reverted).
- **Gap to decide:** no automated test covers `explorer.py --model` (see Scenario Coverage). A small test would close it.

## Verdict

PASS, with the explorer gap and the pending CI evidence recorded above. Every check I ran passed, and every scenario has at least one test or a recorded real run; "Named model" is partial for the explorer clause. If you want that scenario covered strictly, add one test through rag-plan (a task for `explorer.py --model` on a `workcopy`) before the PR.
