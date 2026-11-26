## Task
6.5 Label the sides of `drift.py`'s output `alembic` and `yaml` (CLI labels, passed by the `plumbline-drift` job). Test: a difference prints `only in alembic` or `only in yaml`; the existing drift tests are updated.

## Scope
- **`tools/drift.py`:**
  - `diff(a, b, label_a="a", label_b="b")`: every `only in a`, `only in b` and `a=... b=...` in the output uses the labels. The defaults stay `a` and `b`, so the existing tests and any other caller keep working unchanged.
  - CLI: two optional flags, `--label-a` and `--label-b` (defaults `a` and `b`), passed through to `diff()`.
  - The module docstring's usage line mentions the flags.
- **`.github/workflows/datamodel.yml`:** the drift step becomes `uv run python tools/drift.py "$ALEMBIC_URL" "$GENERATED_URL" --label-a alembic --label-b yaml`.
- Guides read: none apply (datamodel only). I read `drift.py`'s `diff()` and `main()`, `test_drift.py`, and the workflow's drift step.

## Not in scope
- The drift logic, the snapshot queries and the `pg_catalog` handling.
- README and `AGENTS.md` mentions of the drift command: 6.6.

## Files
| File | Change |
| --- | --- |
| `datamodel/tools/drift.py` | labels in `diff()` and the CLI |
| `.github/workflows/datamodel.yml` | pass the two labels |
| `datamodel/tests/test_drift.py` | label tests |

## Tests
- `diff(..., label_a="alembic", label_b="yaml")` on hand-built snapshots: an extra column on `b` prints `column y only in yaml`; an extra constraint on `a` prints `constraint only in alembic`; a table only on `a` prints `only in alembic`; a changed column prints `alembic=[...] yaml=[...]`. Covers the spec scenario "Output names the sides".
- CLI: `main()` run with `sys.argv` set to `drift.py u1 u2 --label-a alembic --label-b yaml` and `snapshot` monkeypatched to return two hand-built snapshots; stdout contains `only in yaml` and the exit code is 1. This runs without Postgres.
- Defaults unchanged: the existing drift tests (`only in a`/`only in b`) still pass as written.

## Checks to run
- `cd datamodel && PYTHONPATH=tools uv run python -m pytest -q -rs tests`
- The workflow YAML parses, and its drift step contains both labels.
- In a throwaway `pgvector/pgvector:pg18` container, as in 4.2: Alembic head vs the generated DDL with the labelled command (expect `no drift`), then the same in a scratch copy with `feedback.rating` made nullable (expect exit 1 and a line saying `alembic=` and `yaml=` with the `rating` column). This is the real output the CI log will show.

## Risks / Questions
- None beyond the flags being optional: `drift.py` with no flags still prints `a` and `b`, so the CLI contract in the README stays true until 6.6 documents the flags.
