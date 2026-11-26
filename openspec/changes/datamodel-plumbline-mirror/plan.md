## Task
6.2 Remove `migrate.py`, `model.lock.json` handling, `migrations/`, `make migrate` and `make check`, and reword the `datamodel` CI job's `make check test`. (`make check` and the CI wording already moved to 6.1.) Test: the suite passes; the spec scenario "Migration tooling absent" holds (no `migrate.py`, `make migrate` or `make check`).

## Scope
- Delete `datamodel/tools/migrate.py` (149 lines): the only code that reads or writes `model.lock.json` or `migrations/`, and the only code with `--allow-destructive`. There are no `model.lock.json` or `migrations/` files left to delete (the sample went in 6.1, and `models/plumbline` never had them).
- `Makefile`: drop `migrate` from `.PHONY`, and delete the `migrate` target with its comment. (`check` went in 6.1.)
- `tools/ddl.py`: the module docstring says "migrate.py diffs two of them. One source for both keeps migrations honest." Reword it to describe what `build_objects()` is now for (the drift job's DDL), nothing else. `build_objects`, `create_table_sql`, `render_full` and `SET_UPDATED_AT` stay: `render_full` uses them, and `SET_UPDATED_AT` goes in 6.4 with the `updated_at_trigger` setting.
- `tools/explorer.template.html`, one line: the callout "DDL, migrations and this page are generated from `domains/**/*.yaml`" becomes "DDL and this page are generated". Migrations are no longer generated.
- A small test file `tests/test_toolkit_scope.py` for the spec scenario "Migration tooling absent": no `tools/migrate.py`, no `migrate:` or `check:` Makefile target, and no `model.lock.json` or `migrations/` under `models/`. 6.3 adds the "Removed field rejected" tests to the same file.
- Guides read: none apply (datamodel only). I grepped `datamodel/`, `AGENTS.md`, `docs/` and `.github/` for every reference to `migrate`, `model.lock.json`, `--allow-destructive` and `make check`.

## Not in scope
- README, `AGENTS.md` (line 159 still lists `migrate`), the overview and ADR 0004: 6.6, the same PR.
- Enums, immutable tables, uuid ids, `sample_rows`, `business_rules` (6.3) and the `model.yaml` settings (6.4).
- `docs/architecture/overview.md` lines about Plumbline's own Alembic migrations, and `backend/migrations/`: unrelated to this tool.

## Files
| File | Change |
| --- | --- |
| `datamodel/tools/migrate.py` | deleted |
| `datamodel/Makefile` | remove the `migrate` target, comment and `.PHONY` entry |
| `datamodel/tools/ddl.py` | docstring only |
| `datamodel/tools/explorer.template.html` | one callout sentence |
| `datamodel/tests/test_toolkit_scope.py` | new |

## Tests
- `test_toolkit_scope.py::test_migration_tooling_absent` covers the spec scenario "Migration tooling absent": `tools/migrate.py` does not exist, the Makefile has no `migrate:` or `check:` target, and nothing named `model.lock.json` or `migrations` exists under `models/`.
- The rest of the suite passes. Nothing else imports `migrate` (checked by grep; the test that imported it was deleted in 6.1).
- `uv run python tools/explorer.py --model models/plumbline` still works and the rebuilt page contains the new callout text.

## Checks to run
- `cd datamodel && PYTHONPATH=tools uv run python -m pytest -q -rs tests`
- `PYTHONPATH=tools uv run python tools/validate.py`
- `PLUMBLINE_EMBEDDING_DIM=8 PYTHONPATH=tools uv run python tools/ddl.py`, and byte-compare `models/plumbline/build/schema.sql` with the previous commit's (unchanged: only a docstring moved).
- `PLUMBLINE_EMBEDDING_DIM=768 PYTHONPATH=tools uv run python tools/explorer.py` and grep the output for the new callout.
- `grep -rn "migrate" datamodel --include=*.py --include=Makefile` finds nothing.
- Postgres-backed run of the suite in a throwaway container is not needed: no test in this task touches Postgres, and nothing Postgres-related changed.

## Risks / Questions
- `AGENTS.md`, the README and ADR 0004 still say `migrate` exists until 6.6. The PR is only coherent after 6.6.
- The test for an absent file and Makefile targets is a guard against the tool coming back by accident, not a behaviour test. It's the only way to cover the spec scenario; say if you'd rather drop that scenario instead.
