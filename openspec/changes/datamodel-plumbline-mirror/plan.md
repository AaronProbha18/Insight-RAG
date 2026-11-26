## Task
6.4 Remove `model.yaml` loading and the `Settings` model; hard-code Plumbline's conventions (`bigserial`, no `public_id` CHECK, no `updated_at` trigger, no comments, Postgres-style names, `bigint` foreign keys); delete `models/plumbline/model.yaml`. Keep `system_columns`. Test: the generated DDL for `models/plumbline` is byte-identical to before this task; a `uuid` foreign key fails validation naming `bigint`.

## Scope
- **`schema.py`:** delete `Settings`.
- **`model.py`:** delete the `Settings` import, `Model.settings`, and the `model.yaml` loading. `con_name(kind, table, cols)` loses its settings argument and always returns Postgres-style names (`<t>_pkey`, `<t>_<c>_fkey`, `<t>_<cols>_key`). Guard: if a model folder contains a `model.yaml`, `load()` raises `ValueError("model.yaml is no longer supported: the toolkit has one set of conventions")`, so a leftover file can't silently look like it applies.
- **`ddl.py`:**
  - No `public_id` CHECK, no comments (the `COMMENT ON` statements, the `comment` fields in the objects dict, and the `q()` quoting helper they used), no `updated_at` trigger, and so no `SET_UPDATED_AT` function or `triggers` entries.
  - `build_objects` and `create_table_sql` lose the settings handling; `render_full` no longer checks for triggers.
- **`validate.py`:** `constraint_names()` yields only the names the DDL generates: primary key, foreign key, unique and the YAML `checks`.
- **Delete** `models/plumbline/model.yaml` and `tests/fixtures/mini/model.yaml`.
- **Keep:** `system_columns`, `unique` groups, YAML `checks`, `indexes` and the review metadata. Plumbline has no indexes today, but the overview names an HNSW index as a later migration and the drift gate already compares indexes, so I'm leaving `indexes` in. Say if you want it gone too.
- Guides read: none apply (datamodel only). I grepped every use of `settings`, `con_name`, `comment` and `SET_UPDATED_AT` in `tools/` and `tests/`.

## Not in scope
- README, `AGENTS.md`, the overview, ADR 0004, the explorer callout text ("The YAML is the model") and the `datamodel-system` delta spec: 6.6.
- Drift output labels (`alembic`/`yaml`): 6.5.
- Any change to `models/plumbline`'s tables.

## Files
| File | Change |
| --- | --- |
| `datamodel/tools/schema.py` | delete `Settings` |
| `datamodel/tools/model.py` | no settings; fixed `con_name`; `model.yaml` guard |
| `datamodel/tools/ddl.py` | fixed conventions; no comments, triggers, `q()`, `SET_UPDATED_AT` |
| `datamodel/tools/validate.py` | `constraint_names()` simplified |
| `datamodel/models/plumbline/model.yaml`, `datamodel/tests/fixtures/mini/model.yaml` | deleted |
| `datamodel/tests/test_toolkit_scope.py` | one new test |

## Tests
- **Byte-identical DDL:** `models/plumbline`'s `schema.sql` is compared with the previous commit's (not a pytest test; run as a check, since it spans two commits), and `test_plumbline_model.py` and `test_conventions.py::test_plumbline_conventions_generated` still pass unchanged. The latter now proves the conventions with no `model.yaml` at all: `bigserial`, `bigint` foreign keys, no `public_id` CHECK, no trigger, no `COMMENT ON`, Postgres-style names.
- **Foreign key type:** the existing `test_non_bigint_foreign_key_rejected` covers "a foreign key that isn't `bigint` fails validation naming `bigint`". A literal `uuid` foreign key no longer loads at all (6.3 removed the type), so the `tasks.md` wording "`uuid` foreign key" is covered by `test_uuid_column_rejected` plus this test.
- **New:** `test_toolkit_scope.py::test_model_yaml_rejected`: a `model.yaml` in a model folder makes `load()` raise, naming `model.yaml`.
- The mini fixture and Plumbline model still validate.

## Checks to run
- `cd datamodel && PYTHONPATH=tools uv run python -m pytest -q -rs tests`
- `validate` and `ddl` on `models/plumbline`, then diff its `schema.sql` against the previous commit's: expect no difference.
- `grep -rn "Settings\|settings\|model.yaml" tools tests` leaves only the guard and its test.
- The Postgres container run isn't needed: the generated SQL is byte-identical, and no test touching Postgres changes.

## Risks / Questions
- **`indexes` kept** (see Scope), a judgment call against your "remove what Plumbline doesn't use" rule.
- **The `model.yaml` guard** is a small addition beyond the task text. Without it a stray `model.yaml` would be silently ignored. Say if you'd rather have no guard.
- **No comments in generated DDL, ever.** Plumbline's tables have none, so nothing changes for the gate, but a future model that wants `COMMENT ON` would need the feature back.
- The task's byte-identical check is a manual diff because it compares two commits.
