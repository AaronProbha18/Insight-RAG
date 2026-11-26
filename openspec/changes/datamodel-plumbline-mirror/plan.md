## Task
4.1 `tools/drift.py <url-a> <url-b>`: a `pg_catalog` snapshot via `psql -At` (columns, `format_type`, not-null, default; `pg_get_constraintdef` sets; name-stripped index defs; triggers), excluding `alembic_version`; readable diff, exit 1 on drift. Test, `pg`-marked on pgvector: same structure with different names and comments -> 0; an extra column -> 1 naming it; a nullability change -> 1.

## Scope
- New `datamodel/tools/drift.py`, standard library only (`argparse`, `json`, `re`, `subprocess`, `sys`), like the other tools.
- **Snapshot** (`snapshot(url) -> dict`): four `psql -X -q -At -v ON_ERROR_STOP=1 -c <sql>` calls, each returning one JSON array (`SELECT coalesce(json_agg(t), '[]') FROM (...) t`), parsed with `json.loads`. JSON avoids any delimiter clash with defaults and check expressions. Scope is schema `public`, ordinary tables (`relkind = 'r'`), excluding `alembic_version`.
  - **columns:** table, column, `format_type(atttypid, atttypmod)`, `attnotnull`, `pg_get_expr(adbin, adrelid)` (default or null).
  - **constraints:** table and `pg_get_constraintdef(oid)`, for every `contype` except `n`. Postgres 18 stores NOT NULL as constraint rows; the columns query already covers nullability, so keeping `n` would only report a nullability change twice.
  - **indexes:** table and `pg_get_indexdef(indexrelid)`, with the index name removed (`CREATE [UNIQUE] INDEX <name> ON` -> `CREATE [UNIQUE] INDEX ON`).
  - **triggers:** table and `pg_get_triggerdef(oid)` for `NOT tgisinternal`, with the trigger name removed.
  - Result shape: `{table: {"columns": {name: [type, notnull, default]}, "constraints": [sorted defs], "indexes": [...], "triggers": [...]}}`.
- **Comparison** (`diff(a, b) -> list[str]`): a pure function. It reports tables only in one side, columns only in one side, a column whose type, nullability or default differs, and constraint, index or trigger definitions present on one side only. Each line names the table, and the column where there is one. Sides are labelled `a` and `b`.
- **Ignored on purpose:** every object name (constraints, indexes, triggers), comments, `alembic_version`, and **column order**. Order is `attnum`, which has no behavioural meaning, so a YAML that lists columns differently from Alembic isn't drift.
- **CLI:** `drift.py <url-a> <url-b>`. It prints `no drift` and exits 0, or prints the diff and exits 1. A failing `psql` call exits 2 with its stderr.
- Guides read: none apply (datamodel only). I read design.md's "Drift gate" section and the spec's drift requirement. `tests/test_postgres.py` is the pattern for the `pg` tests.

## Not in scope
- The CI job (4.2) and docs (5.1).
- Functions' bodies, sequences, extensions, views and other schemas. The spec lists "columns, types, nullability, defaults, constraint definitions, indexes or triggers" only. Trigger functions such as the sample's `trg_*_immutable` are therefore not compared; Plumbline has none.
- Any change to `backend/`.

## Files
| File | Change |
| --- | --- |
| `datamodel/tools/drift.py` | new |
| `datamodel/tests/test_drift.py` | new |

## Tests (`test_drift.py`)
- **No Postgres needed** (these run everywhere, including where `psql` isn't installed): `diff()` on hand-built snapshots. Identical gives `[]`. An extra column on `b` gives a line naming the table and column. A nullability flip gives a line naming the column. A different default gives a line. Renamed constraints and indexes with the same definitions give `[]`. The name-stripping regexes are unit-tested on a sample index and trigger definition.
- **`pg`-marked** (skipped without `DATAMODEL_TEST_DATABASE_URL`; they use the existing `newdb` fixture and `psql` helper from `test_postgres.py`, so they run in the datamodel CI job). The tables use only plain types, so the job's `postgres:16` is enough; nothing in these tests needs the `vector` extension. The gate itself runs on pgvector in 4.2. Two databases each get the same table, built with:
  - different constraint, index and trigger names and different comments -> `drift.py` exits 0;
  - an extra column in one -> exit 1, output names the column;
  - a nullability change in one -> exit 1, output names the column.
  They run `drift.py` as a subprocess, as CI will.

## Checks to run
- `cd datamodel && uv run python -m pytest -q -rs tests`
- `psql` isn't installed on this machine, so the `pg` tests would skip. Docker is installed, so I'll start a throwaway `pgvector/pgvector:pg18` container and put a small `psql` shim (it forwards to `docker exec`) on `PATH` for the run. The shim lives in the scratchpad, not the repo. If Docker isn't running, I'll say the `pg` tests were skipped locally, and CI is then the first real run.
- The sample suite is unchanged: `validate`, `migrate --dry-run` and the 2.x tests still pass.

## Risks / Questions
- **Column order is ignored** (above). It resolves the question left open in 3.1. If you want order treated as drift, it's a one-line change plus a test, but a mirror that must also copy Alembic's column order seems more brittle than useful.
- The queries use `pg_catalog` functions I know exist (`format_type`, `pg_get_expr`, `pg_get_constraintdef`, `pg_get_indexdef`, `pg_get_triggerdef`). Their exact output, such as `bigserial` showing as `nextval('<table>_id_seq'::regclass)`, is UNVERIFIED until it runs against a real Postgres (the container run above).
- The `pg` tests need `psql` on `PATH`, as the existing ones do.
