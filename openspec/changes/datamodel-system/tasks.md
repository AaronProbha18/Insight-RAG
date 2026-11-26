## 1. Port

- [x] 1.1 Copy the owner-supplied `modelkit` into `datamodel/` unchanged (tools, template, tests, sample domains/enums, `0001_baseline.sql`, `model.lock.json`, `.gitignore`); add `datamodel/pyproject.toml` (uv; pydantic, pyyaml; dev pytest; pytest `pythonpath = ["tools"]`) and `uv.lock`; Makefile `PY ?= uv run python`. Test: `make check test` passes (7 prototype tests); `make all` writes `build/schema.sql` and `build/explorer.html`.

## 2. Validator gaps

- [ ] 2.1 Identifier length: table and column names (incl. system columns) over 63 bytes fail, naming table and column. One table-driven test `test_validate.py` covering each validator AC as a (broken model -> expected error) case, including 64- vs 63-byte names. Test: `make test` passes.

## 3. Postgres 16 and explorer

- [ ] 3.1 `tests/test_postgres.py` (marker `pg`, `DATAMODEL_TEST_DATABASE_URL`): immutability triggers (DELETE rejected, UPDATE outside allow-list rejected, allow-listed UPDATE succeeds and advances `updated_at`); baseline + incremental vs fresh build compared via `pg_dump -s`, column order normalised. Test: passes against a local or CI Postgres 16; skipped without the URL.
- [ ] 3.2 Audit `explorer.template.html`: every interpolation of YAML text goes through `esc()`; fix any that don't. Test: by hand - `make explorer`, open `build/explorer.html` locally from disk with a description containing `<script>` and `</script>`; renders, text shown literally, no console errors; recorded in `verification.md`.

## 4. CI and docs

- [ ] 4.1 `.github/workflows/datamodel.yml` (paths `datamodel/**` + itself): uv, Postgres 16 service, `make check test`; decide whether `main.json` lists it as required (path-filtered checks). Test: the PR's CI runs it green; a scratch commit moving YAML without a migration makes it fail (then reverted).
- [ ] 4.2 Docs: `datamodel/README.md` (make and Windows `uv run` commands, opening `build/explorer.html` locally, synthetic-data rule, workflow), architecture overview (what `datamodel/` is and isn't), root README and `AGENTS.md` commands; ADR 0004 status left Proposed pending business tech review. Test: `python scripts/check_links.py` passes.
