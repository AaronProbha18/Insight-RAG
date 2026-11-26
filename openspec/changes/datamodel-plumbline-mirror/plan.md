## Task
2.2 Per-table `system_columns` in `schema.py`, `model.py` and `validate.py`. Test: subset injection; `id`/unknown entries fail validation naming the table.

## Scope
- `schema.py` `Table`:
  - `system_columns: list[str]` defaults to `["public_id", "created_at", "updated_at"]`.
  - `public_id_prefix` becomes `Optional[str] = None`. A table without `public_id` (Plumbline's `chunks`, `feedback`) has no prefix to give.
- `model.py system_columns()`:
  - Injects `id` always, then only the listed optional columns, in today's order.
  - A fully immutable table still drops `updated_at`.
  - The default list gives today's output exactly.
- `validate.py`:
  - An `id` entry, an unknown name, or a duplicate fails with `table <t>: system_columns ...`.
  - `public_id_prefix` is required (and checked for format and uniqueness, as today) only when `public_id` is injected. It's an error when it's set but `public_id` isn't injected.
- `explorer.template.html`: the `public_id <prefix>_…` chip is shown only when a prefix exists. Otherwise the explorer would render `public_id null_…`.
- Guides read: none apply (datamodel only). I read design.md's "Conventions in `model.yaml`" section and the spec's "Per-table system columns" requirement.

## Not in scope
- New types (2.3) and the Plumbline model (3.1).
- `ddl.py`: it already derives everything from the injected columns, and the `public_id` CHECK is already guarded by `"public_id" in columns` since 2.1. No change expected; the tests prove it.

## Files
| File | Change |
| --- | --- |
| `datamodel/tools/schema.py` | `system_columns` field; optional `public_id_prefix` |
| `datamodel/tools/model.py` | `system_columns()` honours the subset |
| `datamodel/tools/validate.py` | entry checks; prefix required only with `public_id` |
| `datamodel/tools/explorer.template.html` | hide the prefix chip when there's no prefix |
| `datamodel/tests/test_conventions.py` | new tests below |

## Tests (on a `workcopy` model)
- Subset: `organization` with `system_columns: [created_at]` gets columns `id` and `created_at` but no `public_id`/`updated_at`. Its DDL has no `public_id` unique or CHECK, and no `tg_organization_updated_at`. Covers *Subset injected*.
- `system_columns: [id]` and `system_columns: [bogus]` each fail validation with a message containing the table name. Covers *Unknown entry*.
- Without `public_id`, the table validates with no prefix, and setting a prefix there is an error.
- The existing suite still passes, and `migrate --dry-run` says "no changes", so the default is unchanged.

## Checks to run
- `cd datamodel && uv run python -m pytest -q -rs tests`
- `uv run python tools/validate.py && uv run python tools/migrate.py --dry-run` prints "no changes"
- Byte compare of the sample's `build/schema.sql` against the previous commit, as in 2.1.

## Risks / Questions
- Making `public_id_prefix` optional isn't spelled out in tasks.md. Without it, though, 3.1's `chunks` and `feedback` would need a dummy prefix, which is exactly the kind of drift the gate exists to catch. So it's a direct consequence of `system_columns`, and I've kept it in this task.
