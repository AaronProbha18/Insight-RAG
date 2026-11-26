## 1. Model selection

- [x] 1.1 Move the sample model (domains, enums, migrations, `model.lock.json`) to `datamodel/models/sample/` with `git mv`. Add `--model` to every tool, and `model_dir()` in `model.py`. Add `MODEL ?= models/sample` to the Makefile; `build/` goes inside the model folder. Update tests and the `conftest.py` `workcopy` fixture. Test: the existing suite passes unchanged in what it asserts; `make check` prints no unmigrated changes; a test runs `ddl.py --model` on a temp copy and checks the output lands in that copy's `build/`.

## 2. Conventions and types

- [x] 2.1 `model.yaml` loading with defaults (`id_type`, `public_id_check`, `updated_at_trigger`, `comments`, `constraint_names`), applied in `ddl.py`; FK type follows `id_type` in `validate.py`. Test: the sample's DDL is byte-identical to before; one test per setting on a temp model.
- [x] 2.2 Per-table `system_columns` in `schema.py`, `model.py` and `validate.py`. Test: subset injection; `id`/unknown entries fail validation naming the table.
- [x] 2.3 Types `char(n)`, `double precision`, `vector(n)`, `vector({embedding_dim})`, filled from `PLUMBLINE_EMBEDDING_DIM` in `ddl.py`/`explorer.py`. Test: filled to `vector(8)`; unset fails naming the variable; `validate` passes without it.

## 3. Plumbline model

- [x] 3.1 `models/plumbline/` with `model.yaml` and `documents`, `chunks`, `answers`, `feedback` YAML matching Alembic head (descriptions from `docs/architecture/overview.md`, synthetic sample rows only). Test: `make validate MODEL=models/plumbline` exits 0; a test asserts the generated tables and their column lists (equality with Alembic is the drift gate, 4.x).

## 4. Drift gate

- [ ] 4.1 `tools/drift.py <url-a> <url-b>`: a `pg_catalog` snapshot via `psql -At` (columns, `format_type`, not-null, default; `pg_get_constraintdef` sets; name-stripped index defs; triggers), excluding `alembic_version`; readable diff, exit 1 on drift. Test, `pg`-marked on pgvector: same structure with different names and comments -> 0; an extra column -> 1 naming it; a nullability change -> 1.
- [ ] 4.2 `plumbline-drift` job in `.github/workflows/datamodel.yml` (pgvector `pg18` service, `PLUMBLINE_EMBEDDING_DIM=8`, paths add `backend/migrations/**`): alembic head into one DB, generated DDL into another, `drift.py`. Not a required check. Test: green on the PR; a scratch commit making one YAML column nullable fails the job, naming the column (then reverted).

## 5. Docs

- [ ] 5.1 ADR 0005 (Proposed, business tech review); `datamodel/README.md` (models, `MODEL=`, `model.yaml`, the drift rule: "change Alembic and the YAML in the same PR"); overview and `AGENTS.md` (backend schema changes now need the YAML too); backend development guide's migration section. Test: `python scripts/check_links.py` passes.
