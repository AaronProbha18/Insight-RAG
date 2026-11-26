Change: `datamodel-system` (issue #21). Verified 2026-11-26 on `feat/21-datamodel-system` at `796dbbc`. Second pass: the first pass failed on scenario gaps, which task 2.2 fixed.

## Checks Run

| Command | Result |
| --- | --- |
| `cd backend && uv run ruff check .` | pass |
| `cd backend && uv run ruff format --check .` | pass (61 files) |
| `docker compose up -d db`, then `cd backend && uv run python -m pytest -q` | pass: 199 passed, 1 skipped |
| `cd frontend && npm run lint` | pass (exit 0) |
| `cd frontend && npm run build` | pass (exit 0) |
| `cd frontend && npm test` | pass: 71 tests |
| `cd datamodel && uv run pytest -q -rs` | pass: 27 passed, 3 skipped (pg: `DATAMODEL_TEST_DATABASE_URL` not set) |
| datamodel suite in a `postgres:16` container with `DATAMODEL_TEST_DATABASE_URL` set | pass: 30 passed |
| `cd datamodel && PYTHONPATH=tools uv run python tools/migrate.py --dry-run` | `no changes` |
| `openspec validate datamodel-system` | valid |
| `openspec validate --all` | 10 passed, 0 failed |
| `python scripts/check_links.py` | links ok (108 files) |

Backend tests ran as `python -m pytest`: `uv run pytest` (and `uv run alembic`) fail locally with `uv trampoline failed to canonicalize script path`, a stale venv launcher from the rename on `main`, not this change.

Recorded in `tasks.md` and not re-run here:
- CI `datamodel` runs 45774816073 (green), 45775860672 (red on purpose: unmigrated YAML), and 45777014136 (green, 26 passed, 0 skipped);
- the explorer check by hand (task 3.2).

## Scenario Coverage

| Spec scenario | Test |
| --- | --- |
| Valid model | `test_validate.py::test_valid_model`, `test_model.py::test_sample_model_is_valid` |
| Foreign key outside depends_on or not at id | `test_validate.py::test_rule_fails[fk outside depends_on]`, `[fk not at id]` |
| Identifier too long | `test_validate.py::test_rule_fails[table name 64 bytes]`, `[column name 64 bytes]`, `test_63_byte_name_has_no_length_error`; `test_model.py::test_long_constraint_name_rejected` |
| Duplicate prefix or rule/question id | `test_validate.py::test_rule_fails[duplicate prefix]`, `[duplicate rule id]` (OQ ids share the same duplicate check) |
| Default on a direct identifier or PHI column | `test_validate.py::test_rule_fails[default on pii column]`, `test_model.py::test_pii_default_rejected` |
| Bad sample row | `test_validate.py::test_rule_fails[sample row unknown column / bad enum / ssn / real email]` |
| Published with an open CMO question | `test_validate.py::test_rule_fails[published with open CMO question]` |
| Answered question without an answer | `test_validate.py::test_rule_fails[answered without answer]` |
| System columns injected | `test_generate.py::test_system_columns_injected` |
| Immutable table trigger | `test_postgres.py::test_immutable_delete_and_update_rejected`, `test_allow_listed_update_advances_updated_at`; `test_model.py::test_immutable_trigger_generated` |
| Enum as CHECK | `test_generate.py::test_enum_is_check_not_type` |
| Migration written | `test_generate.py::test_migration_written` |
| Destructive change | `test_generate.py::test_destructive_change_refused_then_approved`; `test_model.py::test_diff_safe_and_risky` |
| Equivalent to a fresh build | `test_postgres.py::test_migrations_equal_fresh_build` (Postgres 16) |
| Unmigrated change | CI scratch commit `a9c6d5b` failed `make check` (run 45775860672); `test_model.py::test_lock_matches_model` |
| Offline render | Manual check, recorded under task 3.2 in `tasks.md` (manual by design) |

## Eval Results

N/A per eval-plan: a separate `datamodel/` toolchain; nothing in Plumbline's retrieval, prompts or graph changes.

## Deviations

- **Task 2.2 was added after the first verify**, which found 4 scenarios without full tests. The new tests are tests only; no tool changed.
- **CI `pg_dump` is 18.7, not 16.** The server is Postgres 16. `pg_dump` supports older servers, and both sides of the equivalence test use the same binary.
- **Dump normalisation goes beyond column lines.** It also drops `\restrict`/`\unrestrict` lines (`pg_dump` 16.10+) and sorts `COMMENT ON COLUMN` runs. Both are column order or dump noise.
- **The new CI check is not required on `main`** (task 4.1), because it's path-filtered.
- **`main` was merged in** (`2913cf5`; one `AGENTS.md` conflict) instead of rebasing, because force-push is blocked. `main` squash-merges.
- Prototype `README.md`, `docs/DESIGN.md`, `requirements.txt` and `model.yml` were not ported; they're replaced by the datamodel README, ADR 0004, `pyproject.toml` and `datamodel.yml`.
- Follow-up opened: #28 (aligning the YAML model with Plumbline's own schema).

## Verdict

**PASS.**
