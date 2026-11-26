## Task
2.1 `model.yaml` loading with defaults (`id_type`, `public_id_check`, `updated_at_trigger`, `comments`, `constraint_names`), applied in `ddl.py`; FK type follows `id_type` in `validate.py`. Test: the sample's DDL is byte-identical to before; one test per setting on a temp model.

## Scope
- `schema.py`: a strict `Settings` model with the defaults in design.md:
  `id_type: uuid|bigserial = uuid`, `public_id_check = true`, `updated_at_trigger = true`,
  `comments = true`, `constraint_names: generated|postgres = generated`. Unknown keys are rejected.
- `model.py`: `Model.settings` (default `Settings()`), and `load()` reads `<model>/model.yaml` if it exists.
  - `system_columns`/`all_columns` take `id_type`. `bigserial` gives `id bigserial` with no default; `uuid` behaves as today.
  - One shared helper `con_name(settings, kind, table, cols)` returns the name.
    - `generated` keeps today's `pk_`/`fk_`/`uq_`/`ck_..._enum` names.
    - `postgres` gives `<t>_pkey`, `<t>_<c>_fkey`, `<t>_<cols>_key` and `<t>_<c>_check`.
    - YAML `checks` keep their written name in both modes, so the author picks it. Index names stay `ix_...` (the drift gate ignores names).
- `ddl.py build_objects`:
  - The `public_id` CHECK is added only if `public_id_check` is on.
  - The `tg_<t>_updated_at` trigger is added only if `updated_at_trigger` is on.
  - Constraint names come from `con_name`.
  - When `comments: false`, table and column comments are `None`.
- `ddl.py create_table_sql` skips `None` comments. `render_full` emits `set_updated_at()` only if some table has a `tg_*_updated_at` trigger; the sample does, so its output is unchanged.
- `migrate.py` skips `COMMENT ON` for `None` comments (two spots), so `comments: false` models still migrate.
- `validate.py`:
  - A FK column's type must be `uuid` for `id_type: uuid`, or `bigint` for `bigserial`. The error names the expected type.
  - `constraint_names()` uses `con_name` and only yields the CHECK and trigger names that will actually be generated.
- Call sites of `all_columns` in `explorer.py` and `validate.py` pass `m.settings.id_type`.
- Guides read: none apply (datamodel is outside backend/frontend/connector). I read design.md and the spec's "Per-model conventions" requirement.

## Not in scope
- `system_columns` per table (2.2), new types and `vector` (2.3), `models/plumbline` (3.1).
- Verifying Postgres's own check-constraint naming: `<t>_<c>_check` for a column-level enum check is still UNVERIFIED (design.md); task 3.1/4.x confirm it against a real database. The gate ignores names either way.
- The `pgcrypto` extension line stays unconditional (harmless for `bigserial`; dropping it would change the sample's DDL).

## Files
| File | Change |
| --- | --- |
| `datamodel/tools/schema.py` | `Settings` model |
| `datamodel/tools/model.py` | `Model.settings`, read `model.yaml`, `id_type` in `system_columns`/`all_columns`, `con_name()` |
| `datamodel/tools/ddl.py` | apply the four settings; conditional `set_updated_at()`; skip `None` comments |
| `datamodel/tools/migrate.py` | skip `None` comments |
| `datamodel/tools/validate.py` | FK type follows `id_type`; `constraint_names()` via `con_name` |
| `datamodel/tools/explorer.py` | pass `id_type` to `all_columns` |
| `datamodel/tests/test_conventions.py` | new |

## Tests (`test_conventions.py`; each one writes `model.yaml` into a `workcopy` model)
- No `model.yaml`: `render_full` of the sample equals the committed `build/schema.sql` produced before this task (captured from the 1.1 commit with `git show`), and `migrate --dry-run` says "no changes". Covers *No model.yaml keeps today's output*.
- `id_type: bigserial`: `id` is `id bigserial NOT NULL`; the sample's uuid FKs fail validation naming `bigint`; changing them to `bigint` passes. Covers *bigserial keys*.
- `public_id_check: false`: there's no `ck_<t>_public_id`.
- `updated_at_trigger: false`: there's no `tg_*_updated_at` and no `set_updated_at()` in the DDL. (The sample's immutable-table triggers stay.)
- `comments: false`: `COMMENT ON` doesn't appear in the DDL.
- `constraint_names: postgres`: `organization_pkey`, `<t>_<c>_fkey` and `<t>_<c>_key` appear, and no `pk_`/`fk_`/`uq_` names are left. Together with the three tests above, this covers *Conventions switched off*.
- An unknown key in `model.yaml` fails to load, naming the key.

## Checks to run
- `cd datamodel && uv run python -m pytest -q -rs tests` (the targets of `make check test`, run directly because `make` isn't installed in this shell)
- `uv run python tools/validate.py && uv run python tools/migrate.py --dry-run` prints "no changes"

## Risks / Questions
- `validate.py`'s Postgres identifier-length check now runs on `postgres`-style names too. That's correct, but it's new coverage.
- I'm assuming that, in `postgres` mode, YAML `checks` keep the author-written name rather than becoming `<t>_check`. Design.md marks Postgres's own unnamed-check naming as UNVERIFIED and says names "only matter for readable DDL", so keeping the written name is the safe, explicit option.
