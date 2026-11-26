## Task
2.3 Types `char(n)`, `double precision`, `vector(n)`, `vector({embedding_dim})`, filled from `PLUMBLINE_EMBEDDING_DIM` in `ddl.py`/`explorer.py`. Test: filled to `vector(8)`; unset fails naming the variable; `validate` passes without it.

## Scope
- `schema.py` `TYPE` regex: add `char\(\d{1,4}\)`, `double precision`, `vector\(\d{1,5}\)` and the literal `vector\(\{embedding_dim\}\)`.
- `model.py`: `column_type(c) -> str` replaces `{embedding_dim}` with `PLUMBLINE_EMBEDDING_DIM`.
  - It fails with `ValueError("PLUMBLINE_EMBEDDING_DIM must be set to a positive integer ...")` when the variable is unset or isn't a positive integer, matching backend migration 0001's own check.
  - It only touches the environment when the placeholder is present, so the sample never needs the variable.
- `ddl.py build_objects` and `explorer.py payload` use `column_type(c)` instead of `c.type`. `validate.py` keeps the raw `c.type`, so it passes without the variable.
- `migrate.py _parts`: the regex takes the type as a single token, so `double precision` would be split apart. Matching `double precision` as one type keeps type-change diffs right. A unit test on `_parts` covers it.
- Guides read: none apply (datamodel only). I also read the spec's new-types requirement and backend `migrations/versions/0001_initial.py` for the variable's rule.

## Not in scope
- `CREATE EXTENSION vector` in generated DDL. Design.md has the drift job create the extension before `psql -f` (4.2), and the sample doesn't need it.
- The Plumbline model (3.1).

## Files
| File | Change |
| --- | --- |
| `datamodel/tools/schema.py` | widen `TYPE` |
| `datamodel/tools/model.py` | `column_type()` |
| `datamodel/tools/ddl.py`, `explorer.py` | use `column_type()` |
| `datamodel/tools/migrate.py` | `_parts` handles `double precision` |
| `datamodel/tests/test_conventions.py` | tests below |

## Tests
- A `workcopy` table gets columns `vector({embedding_dim})`, `char(64)` and `double precision`. With `monkeypatch.setenv("PLUMBLINE_EMBEDDING_DIM", "8")`, the DDL has `vector(8)`, `char(64)` and `double precision`. Covers *Embedding dimension filled*.
- With the variable deleted, `build_objects` raises naming `PLUMBLINE_EMBEDDING_DIM`; `"0"` raises too. Covers *Dimension missing*.
- With the variable unset, `validate.validate()` returns `[]`.
- `migrate._parts("x double precision NOT NULL")` gives the type `double precision` with not-null set.
- The sample is unchanged: the suite passes, `migrate --dry-run` says "no changes", and the byte compare matches.

## Checks to run
- `cd datamodel && uv run python -m pytest -q -rs tests`
- `uv run python tools/validate.py && uv run python tools/migrate.py --dry-run`
- Byte compare of the sample's `schema.sql` against the previous commit.

## Risks / Questions
- `explorer.py` will now need the variable for any model that has `vector({embedding_dim})`, so `make explorer MODEL=models/plumbline` without it fails with the named error. That's what the spec says for generation; the toolkit README notes it in 5.1.
