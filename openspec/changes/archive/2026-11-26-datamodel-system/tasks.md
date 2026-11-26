## 1. Port

- [x] 1.1 Copy the owner-supplied `modelkit` into `datamodel/` unchanged (tools, template, tests, sample domains/enums, `0001_baseline.sql`, `model.lock.json`, `.gitignore`); add `datamodel/pyproject.toml` (uv; pydantic, pyyaml; dev pytest; pytest `pythonpath = ["tools"]`) and `uv.lock`; Makefile `PY ?= uv run python`. Test: `make check test` passes (7 prototype tests); `make all` writes `build/schema.sql` and `build/explorer.html`.

## 2. Validator gaps

- [x] 2.1 Identifier length: table and column names (incl. system columns) over 63 bytes fail, naming table and column. One table-driven test `test_validate.py` covering each validator AC as a (broken model -> expected error) case, including 64- vs 63-byte names. Test: `make test` passes.

- [x] 2.2 Generator and migrate CLI tests (gaps found by verify): `tests/test_generate.py` asserts injected system columns (and no `updated_at` on a fully immutable table), enum columns as `ck_<t>_<c>_enum` CHECKs with no `CREATE TYPE`, `migrate.py --name x` on a temp copy writing `NNNN_x.sql` plus an updated lock, and a destructive change exiting 2 without `--allow-destructive` and writing the `RISKY, approved` header with it. Test: `make test` passes without Postgres.

## 3. Postgres 16 and explorer

- [x] 3.1 `tests/test_postgres.py` (marker `pg`, `DATAMODEL_TEST_DATABASE_URL`): immutability triggers (DELETE rejected, UPDATE outside allow-list rejected, allow-listed UPDATE succeeds and advances `updated_at`); baseline + incremental vs fresh build compared via `pg_dump -s`, column order normalised. Test: passes against a local or CI Postgres 16; skipped without the URL.
- [x] 3.2 Audit `explorer.template.html`: every interpolation of YAML text goes through `esc()`; fix any that don't. Test: by hand - `make explorer`, open `build/explorer.html` locally from disk with a description containing `<script>` and `</script>`; renders, text shown literally, no console errors; recorded in `verification.md`.
  - Result 2026-11-26: audit found every YAML-text interpolation already escaped (sidebar included); no fix. Description `Test <script>alert(1)</script> & "quotes" </script><!--`: built payload holds `<\/script>`/`<\!--`; opened from disk (`file:///`) by the owner: no alert, text literal, console clear.

## 4. CI and docs

- [x] 4.1 `.github/workflows/datamodel.yml` (paths `datamodel/**` + itself): uv, Postgres 16 service, `make check test`; decide whether `main.json` lists it as required (path-filtered checks). Test: the PR's CI runs it green; a scratch commit moving YAML without a migration makes it fail (then reverted).
  - Result 2026-11-26: not a required check (path-filtered; a required one would block other PRs). Run 45774816073 (26e25a0) green, 26 passed, 0 skipped. Scratch a9c6d5b (YAML column, no migration): run 45775860672 failed at `make check` ("model has unmigrated changes"); reverted in 951deaf. Run 45777014136 (2913cf5) green, 26 passed. CI pg_dump is 18.7 against the Postgres 16 server; both sides of the equivalence test use the same binary.
- [x] 4.2 Docs: `datamodel/README.md` (make and Windows `uv run` commands, opening `build/explorer.html` locally, synthetic-data rule, workflow), architecture overview (what `datamodel/` is and isn't), root README and `AGENTS.md` commands; ADR 0004 status left Proposed pending business tech review. Test: `python scripts/check_links.py` passes.
