## Task
1.1 Move the sample model (domains, enums, migrations, `model.lock.json`) to `datamodel/models/sample/` with `git mv`. Add `--model` to every tool, and `model_dir()` in `model.py`. Add `MODEL ?= models/sample` to the Makefile; `build/` goes inside the model folder. Update tests and the `conftest.py` `workcopy` fixture. Test: the existing suite passes unchanged in what it asserts; `make check` prints no unmigrated changes; a test runs `ddl.py --model` on a temp copy and checks the output lands in that copy's `build/`.

## Scope
- `git mv datamodel/{domains,enums,migrations,model.lock.json} datamodel/models/sample/`.
- `model.py`: `ROOT` stays the toolkit root. Add `DEFAULT_MODEL = "models/sample"` and
  `model_dir(name=DEFAULT_MODEL) -> ROOT / name` (absolute paths pass through, because `ROOT / abs` returns `abs`).
  `load(root=None)` uses `model_dir()` when `root` is None.
- Each tool's `main()` takes `--model` (default `DEFAULT_MODEL`), loads from `model_dir(args.model)`, and writes to
  `<model>/build/`. `migrate.py`: `LOCK`/`MIG` become per-model paths derived inside `main()`.
  `validate.py`, `ddl.py` and `explorer.py` gain a small argparse (`migrate.py` already has one).
  `explorer.py` still reads its template from `ROOT/tools`.
- Makefile: `MODEL ?= models/sample`, passed as `--model $(MODEL)` to every tool target; `clean` removes `$(MODEL)/build`.
- Guides read: none apply (`datamodel/` is not backend/frontend/connector). I read ADR 0004 via design.md and `datamodel/Makefile`.

## Not in scope
- `model.yaml`, `system_columns`, new types (2.x), the Plumbline model (3.1), drift (4.x), README/ADR/docs (5.1).
- `.github/workflows/datamodel.yml`: it runs `make check test` with the default MODEL, so it needs no change now.

## Files
| File | Change |
| --- | --- |
| `datamodel/{domains,enums,migrations,model.lock.json}` | `git mv` to `datamodel/models/sample/` |
| `datamodel/tools/model.py` | `DEFAULT_MODEL`, `model_dir()`, default `load()` root |
| `datamodel/tools/validate.py`, `ddl.py`, `explorer.py` | `--model` arg; output to `<model>/build/` |
| `datamodel/tools/migrate.py` | `--model` arg; lock/migrations paths per model |
| `datamodel/Makefile` | `MODEL ?= models/sample`; pass `--model`; `clean` path |
| `datamodel/tests/conftest.py` | `workcopy` copies `tools/` to `work/tools` and `models/sample` to `work/models/sample`; returns the **model dir**; `run_tool` resolves tools from `work.parent.parent / "tools"`, with cwd at the toolkit copy |
| `datamodel/tests/test_postgres.py` | `migrated()` default root becomes `model_dir()` |
| `datamodel/tests/test_generate.py` | new `test_ddl_model_flag_writes_into_model_build` |

The `workcopy` return value is the model dir, so existing paths such as `workcopy / "domains/..."` and `work / "build/schema.sql"` keep working unchanged.

## Tests
- Existing suite unchanged in what it asserts (spec: sample model behaviour preserved).
- New: copy `models/sample` to a tmp dir, run `ddl.py --model <tmp>`, assert `<tmp>/build/schema.sql` exists and equals the default model's DDL. This covers the "model selection" scenario.

## Checks to run
- `cd datamodel && make check test` (`pg` tests run if the local pgvector/PG URL is set, otherwise they're skipped; I report which)
- `cd datamodel && make all` (artifacts land in `models/sample/build/`)

## Risks / Questions
- Before this change, `make ddl` wrote to `datamodel/build/`. A stale `datamodel/build/` stays (gitignored), and `make clean` no longer removes it. Harmless.
- ADR 0004 and README mention the old paths; they're fixed in 5.1, not here.
