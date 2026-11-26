## Task
2.2 Generator and migrate CLI tests (gaps found by verify): `tests/test_generate.py` asserts injected system columns (and no `updated_at` on a fully immutable table), enum columns as `ck_<t>_<c>_enum` CHECKs with no `CREATE TYPE`, `migrate.py --name x` on a temp copy writing `NNNN_x.sql` plus an updated lock, and a destructive change exiting 2 without `--allow-destructive` and writing the `RISKY, approved` header with it. Test: `make test` passes without Postgres.

## Scope
- New `tests/test_generate.py`; tests only, no tool changes.
- Reuse: the temp-copy-and-run-tools code already in `test_postgres.py::test_migrations_equal_fresh_build` moves into `tests/conftest.py`:
  - a `workcopy` fixture that copies domains, enums, tools, migrations and the lock to `tmp_path`;
  - a `run_tool(work, script, *args)` helper that returns the `CompletedProcess`.

  `test_postgres.py` then uses them too, so the copy logic exists once.
- Guides read: none apply.

## Not in scope
- Any change to `tools/`. If a test exposes a generator bug, stop and report.
- The backend `test_invalid_request_422` skip problem (pre-existing; its own issue).

## Files
| File | Change |
| --- | --- |
| `datamodel/tests/conftest.py` | new: `workcopy` fixture, `run_tool` helper |
| `datamodel/tests/test_postgres.py` | use `workcopy`/`run_tool` instead of the inline copy and subprocess |
| `datamodel/tests/test_generate.py` | new, 4 tests |

## Tests
- **System columns injected:** `ddl.build_objects(load())`.
  - `organization` columns start with `id`, `public_id` and include `created_at`, `updated_at`.
  - `id` is `uuid ... DEFAULT gen_random_uuid()` with `pk_organization` as the primary key.
  - `ck_organization_public_id` matches `^org_[0-9A-Za-z]+$`.
  - `public_id` is unique.
  - `person_consent_grant` (fully immutable) has no `updated_at`; `membership_order` (has `mutable_columns`) keeps it.
  - No sample YAML declares a system column.
- **Enum as CHECK:** `organization.status` has `ck_organization_status_enum` = `CHECK (status IN ('ACTIVE', 'INACTIVE'))` or the generator's exact form, read from `ddl.py` when writing the test. `render_full` contains no `CREATE TYPE`.
- **Migration written:** in `workcopy`, add a nullable column to the YAML, run `migrate.py --name add_x`, and assert:
  - exit 0;
  - `migrations/0002_add_x.sql` exists, wrapped in `BEGIN;`/`COMMIT;`, with `ADD COLUMN`;
  - the lock now contains the column, and a second `--dry-run` prints `no changes`.
- **Destructive change:** in `workcopy`, remove a column from the YAML.
  - `migrate.py --name drop_x` exits 2, prints `REFUSED`, and writes no file or lock.
  - With `--allow-destructive`, it exits 0, and the file has `-- RISKY, approved with --allow-destructive:` and `DESTRUCTIVE`.

## Checks to run
- `cd datamodel && uv run pytest -q -rs` -> 23 + 4 new passed, 3 pg skipped.
- The pg tests again in Docker (`postgres:16`, same setup as 3.1), since `test_postgres.py` is refactored -> 3 passed.

## Risks / Questions
- Removing a column: pick one not referenced by indexes, unique or checks (e.g. `org_clinic_location.phone`), so validate still passes and only migrate refuses.
