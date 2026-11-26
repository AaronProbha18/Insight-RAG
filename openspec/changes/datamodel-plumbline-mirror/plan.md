## Task
6.1 Add a small fixture model built inside `tests/` (two Plumbline-shaped tables with a foreign key, a check and `system_columns`), and rebuild `conftest.py`'s `workcopy` on it. Re-point the tests that don't depend on removed features. Delete `models/sample/` (with its `build/`), and delete the tests for migrations, enums and immutable tables, which 6.2 and 6.3 remove from the code. `DEFAULT_MODEL` and the Makefile default become `models/plumbline`. Test: the remaining datamodel suite passes; `validate` and `ddl` run on `models/plumbline` with no `--model`.

## Scope
- **Fixture:** `datamodel/tests/fixtures/mini/`, Plumbline-shaped so later tasks barely touch it. It has a `model.yaml` with Plumbline's five settings (`id_type: bigserial`, `public_id_check: false`, `updated_at_trigger: false`, `comments: false`, `constraint_names: postgres`), which 6.4 deletes. No enums, immutable tables, uuid ids, `sample_rows` or `business_rules`.
  - Domain `core` (`load_order: 10`) with tables `account` (prefix `acc`; columns `display_name text` and `first_name text`, `pii: indirect`) and `note` (prefix `nte`, `system_columns: [public_id, created_at]`; `account_id bigint` referencing `account.id`, `on_delete: cascade`; `body text`; one table check `ck_note_body`, `body <> ''`).
  - Domain `audit` (`load_order: 20`, `depends_on: []`) with table `event` (prefix `evt`, `system_columns: [created_at]`; `kind text`).
  - Both domains' tables are `status: published`, so the open-CMO-question rule is testable.
- **`conftest.py`:** `ROOT` and `run_tool` as today. `workcopy` copies `tools/` and `tests/fixtures/mini/` into a temp dir (`dm/models/mini`) and returns the model folder, so existing `workcopy / ...` paths keep their shape. A `MINI` path constant for tests that only read the fixture. The Postgres helpers (`URL`, `_db_url`, `psql`, `apply`, `newdb`) stay in `test_postgres.py`, where `test_drift.py` imports them today.
- **Delete** `models/sample/` (16 tracked files) and its ignored `build/`.
- **Default model:** `model.py` `DEFAULT_MODEL = "models/plumbline"`; Makefile `MODEL ?= models/plumbline`.
- **Makefile and CI (moved here from 6.2):** with `models/plumbline` as default there is no lock file for `make check` to compare, so remove the `check` target now, and the `datamodel` job runs `make validate test` instead of `make check test`. 6.2 still removes `migrate.py` and `make migrate`.
- Guides read: none apply (datamodel only). I read the five sample-dependent test files and `conftest.py`.

## Tests: what happens to each file
| File | Action |
| --- | --- |
| `test_model.py` | Keep `valid` (on the fixture), `long_constraint_name_rejected`, `pii_default_rejected`, `unknown_fk_rejected`, re-pointed to `MINI`. Delete `immutable_trigger_generated`, `diff_safe_and_risky`, `lock_matches_model`. |
| `test_validate.py` | Re-point to fixture tables. Keep: fk outside `depends_on`, fk not at `id`, 64-byte table and column names, duplicate prefix, default on a pii column, published with an open CMO question, answered without an answer, and the 63-byte cases. Delete: duplicate rule id, the four `sample row` cases. |
| `test_generate.py` | Keep `system_columns_injected` rewritten for the fixture (bigserial `id`, `public_id` unique, `created_at` last, `event` has no `public_id`) and `ddl_model_flag_writes_into_model_build`. Delete `enum_is_check_not_type`, `migration_written`, `destructive_change_refused_then_approved`. |
| `test_conventions.py` | Keep the three `system_columns` tests and the four type tests, edited to use `account.yaml`. Delete `no_model_yaml_keeps_lock`, `bigserial_ids_and_bigint_fks`, `public_id_check_off`, `updated_at_trigger_off`, `comments_off`, `unknown_key_rejected` (settings, removed in 6.4). Replace `postgres_constraint_names` with `plumbline_conventions_generated`. |
| `test_postgres.py` | Keep the helpers. Delete the three tests (immutability, allow-listed update, migrations equal a fresh build). Add `test_fixture_ddl_applies`: the fixture's generated DDL loads into a real Postgres, and the tables exist. |
| `test_plumbline_model.py`, `test_drift.py` | Unchanged. |

New spec-scenario tests:
- **Plumbline conventions generated:** `test_conventions.py::test_plumbline_conventions_generated`. The fixture DDL has `id bigserial`, `bigint` foreign keys, no `public_id` CHECK, no `updated_at` trigger or `set_updated_at()`, no `COMMENT ON`, and `_pkey`/`_fkey`/`_key` names with no `pk_`/`fk_`/`uq_`.
- **Default is Plumbline's model:** a test runs `validate.py` and `ddl.py` as subprocesses with no `--model` and `PLUMBLINE_EMBEDDING_DIM=8`, in a copy of the toolkit, and checks `models/plumbline/build/schema.sql` appears. `workcopy`-style, so the repo's own `build/` isn't touched.

## Checks to run
- `cd datamodel && PYTHONPATH=tools uv run python -m pytest -q -rs tests`
- `PYTHONPATH=tools uv run python tools/validate.py` (no `--model`), and `PLUMBLINE_EMBEDDING_DIM=8 uv run python tools/ddl.py` (no `--model`).
- The suite once more inside a throwaway `pgvector/pgvector:pg18` container with `DATAMODEL_TEST_DATABASE_URL` set, so the Postgres tests really run (as in 4.1).
- The workflow YAML parses (`yaml.safe_load`), and `make validate test` is the new `datamodel` job command.
- `git diff --stat` shows only the files in the table below.

## Files
| File | Change |
| --- | --- |
| `datamodel/models/sample/**` | deleted (16 tracked files) |
| `datamodel/tests/fixtures/mini/**` | new (`model.yaml`, 2 domains, 3 tables) |
| `datamodel/tests/conftest.py` | `workcopy` on the fixture; `MINI` |
| `datamodel/tests/test_model.py`, `test_validate.py`, `test_generate.py`, `test_conventions.py`, `test_postgres.py` | per the table above |
| `datamodel/tools/model.py` | `DEFAULT_MODEL` |
| `datamodel/Makefile` | `MODEL ?= models/plumbline`; remove `check` |
| `.github/workflows/datamodel.yml` | `make validate test` |

## Not in scope
- Removing `migrate.py`, `make migrate`, enums, immutable tables, uuid ids, `sample_rows`, `business_rules`, and the settings: 6.2 to 6.4. Their code stays, untested, until then.
- README, `AGENTS.md`, overview and ADR text that mention the sample or `make check`: 6.6, the same PR (AGENTS.md rule 7).
- The stale `datamodel/build/` folder (ignored, pre-#28 leftover).

## Risks / Questions
- **Coverage dips until 6.2 to 6.4:** migrations, enums, immutable tables, uuid ids and the `model.yaml` settings have no tests between this task and their removal. That's the point of removing them, but each intermediate commit leaves untested code; the PR is only green at the end of 6.6.
- **`make check` and the CI wording move from 6.2 to 6.1** because the default model changes. I'll note it when I tick 6.1; `tasks.md` 6.2 keeps its text, and 6.2's "reword CI" part becomes a no-op.
- **The fixture is also a sample**, just a tiny one in `tests/`. It's what you approved ("small fixture inside tests/"); say so if you'd rather build the YAML in `conftest.py` from strings.
- **ADR 0004** (Accepted) still describes the sample until 6.6 rewrites it.
