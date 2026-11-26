## 1. Model selection

- [x] 1.1 Move the sample model (domains, enums, migrations, `model.lock.json`) to `datamodel/models/sample/` with `git mv`. Add `--model` to every tool, and `model_dir()` in `model.py`. Add `MODEL ?= models/sample` to the Makefile; `build/` goes inside the model folder. Update tests and the `conftest.py` `workcopy` fixture. Test: the existing suite passes unchanged in what it asserts; `make check` prints no unmigrated changes; a test runs `ddl.py --model` on a temp copy and checks the output lands in that copy's `build/`. (The sample part is superseded by 6.1; `--model` stays.)

## 2. Conventions and types

- [x] 2.1 `model.yaml` loading with defaults (`id_type`, `public_id_check`, `updated_at_trigger`, `comments`, `constraint_names`), applied in `ddl.py`; FK type follows `id_type` in `validate.py`. Test: the sample's DDL is byte-identical to before; one test per setting on a temp model. (Superseded by 6.4.)
- [x] 2.2 Per-table `system_columns` in `schema.py`, `model.py` and `validate.py`. Test: subset injection; `id`/unknown entries fail validation naming the table.
- [x] 2.3 Types `char(n)`, `double precision`, `vector(n)`, `vector({embedding_dim})`, filled from `PLUMBLINE_EMBEDDING_DIM` in `ddl.py`/`explorer.py`. Test: filled to `vector(8)`; unset fails naming the variable; `validate` passes without it.

## 3. Plumbline model

- [x] 3.1 `models/plumbline/` with `model.yaml` and `documents`, `chunks`, `answers`, `feedback` YAML matching Alembic head (descriptions from `docs/architecture/overview.md`, synthetic sample rows only). Test: `make validate MODEL=models/plumbline` exits 0; a test asserts the generated tables and their column lists (equality with Alembic is the drift gate, 4.x). (`model.yaml` and `sample_rows` are removed by 6.3 and 6.4.)

## 4. Drift gate

- [x] 4.1 `tools/drift.py <url-a> <url-b>`: a `pg_catalog` snapshot via `psql -At` (columns, `format_type`, not-null, default; `pg_get_constraintdef` sets; name-stripped index defs; triggers), excluding `alembic_version`; readable diff, exit 1 on drift. Test, `pg`-marked on pgvector: same structure with different names and comments -> 0; an extra column -> 1 naming it; a nullability change -> 1.
- [x] 4.2 `plumbline-drift` job in `.github/workflows/datamodel.yml` (pgvector `pg18` service, `PLUMBLINE_EMBEDDING_DIM=8`, paths add `backend/migrations/**`): alembic head into one DB, generated DDL into another, `drift.py`. Not a required check. Test: green on the PR; a scratch commit making one YAML column nullable fails the job, naming the column (then reverted).

## 5. Docs

- [x] 5.1 Extend ADR 0004 (business tech review) with the mirror decision, `models/` layout and conventions; `datamodel/README.md` (models, `MODEL=`, `model.yaml`, the drift rule: "change Alembic and the YAML in the same PR"); overview and `AGENTS.md` (backend schema changes now need the YAML too); backend development guide's migration section. Test: `python scripts/check_links.py` passes.

## 6. Trim the toolkit to what Plumbline uses

Order matters: enums, immutable tables and uuid ids can't be removed while the sample, which uses them, still exists, so 6.1 goes first.

- [ ] 6.1 Add a small fixture model built inside `tests/` (two Plumbline-shaped tables with a foreign key, a check and `system_columns`), and rebuild `conftest.py`'s `workcopy` on it. Re-point the tests that don't depend on removed features. Delete `models/sample/` (with its `build/`), and delete the tests for migrations, enums and immutable tables, which 6.2 and 6.3 remove from the code. `DEFAULT_MODEL` and the Makefile default become `models/plumbline`. Test: the remaining datamodel suite passes; `validate` and `ddl` run on `models/plumbline` with no `--model`.
- [ ] 6.2 Remove `migrate.py`, `model.lock.json` handling, `migrations/`, `make migrate` and `make check`, and reword the `datamodel` CI job's `make check test`. Test: the suite passes; the spec scenario "Migration tooling absent" holds (no `migrate.py`, `make migrate` or `make check`).
- [ ] 6.3 Remove enums (the `enums/` folder, `EnumDef`, the column `enum` field, the enum checks), immutable tables (`immutable`, `mutable_columns`, trigger generation, validation), uuid ids, `sample_rows` (and the SSN and email checks) and `business_rules`, from `schema.py`, `model.py`, `validate.py`, `ddl.py`, `explorer.py` and the explorer template; remove `sample_rows` from the four Plumbline tables. Test: a table YAML that sets `immutable`, `sample_rows`, `business_rules` or a column `enum` fails to load naming the field; the Plumbline model still validates and generates.
- [ ] 6.4 Remove `model.yaml` loading and the `Settings` model; hard-code Plumbline's conventions (`bigserial`, no `public_id` CHECK, no `updated_at` trigger, no comments, Postgres-style names, `bigint` foreign keys); delete `models/plumbline/model.yaml`. Keep `system_columns`. Test: the generated DDL for `models/plumbline` is byte-identical to before this task; a `uuid` foreign key fails validation naming `bigint`.
- [ ] 6.5 Label the sides of `drift.py`'s output `alembic` and `yaml` (CLI labels, passed by the `plumbline-drift` job). Test: a difference prints `only in alembic` or `only in yaml`; the existing drift tests are updated.
- [ ] 6.6 Docs: rewrite ADR 0004 in place with a revision note (Alembic is the source, the YAML mirrors it, the toolkit is limited to what that needs); update `datamodel/README.md`, `AGENTS.md` and `docs/architecture/overview.md`; create the delta spec `specs/datamodel-system/spec.md` in this change (REMOVED: Migrations; MODIFIED: validator, DDL generation and explorer requirements, plus CI gate to match). Test: `python scripts/check_links.py` and `openspec validate datamodel-plumbline-mirror` pass.
