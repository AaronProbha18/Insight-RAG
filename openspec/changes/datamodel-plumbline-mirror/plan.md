## Task
6.3 Remove enums (the `enums/` folder, `EnumDef`, the column `enum` field, the enum checks), immutable tables (`immutable`, `mutable_columns`, trigger generation, validation), uuid ids, `sample_rows` (and the SSN and email checks) and `business_rules`, from `schema.py`, `model.py`, `validate.py`, `ddl.py`, `explorer.py` and the explorer template; remove `sample_rows` from the four Plumbline tables. Test: a table YAML that sets `immutable`, `sample_rows`, `business_rules` or a column `enum` fails to load naming the field; the Plumbline model still validates and generates.

## Scope
- **`schema.py`:**
  - Remove `EnumDef`, `Rule`, the column `enum` field, and the table fields `immutable`, `mutable_columns`, `business_rules` and `sample_rows`. The models forbid unknown fields, so a YAML that still sets one fails to load and names it.
  - Remove `Settings.id_type`: with uuid ids gone there is one id type, so the setting has no meaning. The other four settings go in 6.4.
  - Slim the column `TYPE` regex to what Plumbline uses: `text`, `int`, `bigint`, `bool`, `jsonb`, `timestamptz`, `char(n)`, `double precision`, `vector(n)`, `vector({embedding_dim})`. That drops `uuid`, `date`, `time`, `inet`, `numeric(p,s)` and `varchar(n)`, which Plumbline's four tables don't use (your rule: "if not used in Plumbline, remove it"). Say if you'd rather keep the other basic types.
- **`model.py`:** drop the `enums` field and the `enums/` loading; `system_columns()` always injects `id bigserial` (no uuid branch, no `id_type` parameter); `updated_at` is injected whenever it is in `system_columns` (no immutable exception); drop the `enum` name kind from `con_name()`. Update the module docstring.
- **`validate.py`:** drop the enum, immutable, `mutable_columns`, `business_rules` and `sample_rows` checks, the SSN and email checks, the enum-value rules, the uuid/`id_type` branch (a foreign key must be `bigint`, always), and the `enums` count in the summary line. `constraint_names()` stops yielding enum-check and immutable-trigger names. Keep: names, prefixes, `system_columns`, foreign keys, `unique`, indexes, PII default rule, open-question rules, the CMO-published rule, identifier lengths.
- **`ddl.py`:** drop the enum CHECK generation and the immutable trigger block, and the `CREATE EXTENSION IF NOT EXISTS pgcrypto;` line. It only existed for `gen_random_uuid()`, which nothing generates now. **This changes the generated DDL by that one line**; Plumbline's schema never used pgcrypto.
- **`explorer.py` and the template:** remove the enums, immutable, rules and samples fields from the payload; in the template, remove the Enums nav item and page, the Enums, Sample rows and Rules stat tiles, the Rules and Samples tabs, the immutable chip and the enum pill. Open questions, columns, constraints and relationships stay.
- **`model.yaml` files** (the mini fixture and `models/plumbline`): remove the `id_type: bigserial` line, since the key no longer exists.
- **`models/plumbline`:** remove the `sample_rows` blocks from `documents`, `answers` and `feedback`.
- Guides read: none apply (datamodel only). I grepped every use of each removed field in `tools/` and the template.

## Not in scope
- The other four settings (`public_id_check`, `updated_at_trigger`, `comments`, `constraint_names`) and `model.yaml`: 6.4.
- `SET_UPDATED_AT` and the `updated_at` trigger generation: 6.4, with its setting.
- README, `AGENTS.md`, the overview, ADR 0004 and the `datamodel-system` delta spec: 6.6.
- Opening the explorer in a browser (not possible here); see Risks.

## Files
| File | Change |
| --- | --- |
| `datamodel/tools/schema.py` | remove the fields, classes and types above |
| `datamodel/tools/model.py` | no enums, no uuid, no immutable branch |
| `datamodel/tools/validate.py` | drop the removed-feature checks |
| `datamodel/tools/ddl.py` | no enum CHECK, no immutable trigger, no pgcrypto |
| `datamodel/tools/explorer.py`, `explorer.template.html` | drop the removed sections |
| `datamodel/tests/fixtures/mini/model.yaml`, `datamodel/models/plumbline/model.yaml` | remove `id_type` |
| `datamodel/models/plumbline/domains/rag/tables/{documents,answers,feedback}.yaml` | remove `sample_rows` |
| `datamodel/tests/test_toolkit_scope.py` | new tests below |
| `datamodel/tests/test_conventions.py` | the uuid-foreign-key test uses an `int` foreign key |

## Tests
- `test_toolkit_scope.py` (spec scenario "Removed field rejected"): for each of `immutable`, `mutable_columns`, `business_rules`, `sample_rows` (table fields) and `enum` (a column field), edit the mini fixture in a temp copy and assert `load()` raises a `ValidationError` naming the field. Plus a column typed `uuid` is rejected as an unsupported type.
- The explorer payload built from the mini fixture has no `enums`, `rules`, `samples` or `immutable` keys.
- `test_conventions.py`: `test_uuid_foreign_key_rejected` now uses an `int` foreign key, because a `uuid` column no longer loads; the validator still rejects it and names `bigint`.
- The rest of the suite passes; `test_plumbline_model.py` still shows the four tables and their columns.

## Checks to run
- `cd datamodel && PYTHONPATH=tools uv run python -m pytest -q -rs tests`
- `validate` and `ddl` on `models/plumbline`; diff its `schema.sql` against the previous commit's. Expect exactly one removed line (`CREATE EXTENSION IF NOT EXISTS pgcrypto;`).
- Explorer: regenerate for `models/plumbline`, extract the page's `<script>` and run `node --check` on it, and grep the page for `Enums`, `Sample rows` and `immutable` (expect none).
- In a throwaway `pgvector/pgvector:pg18` container, as in 4.2: the suite with Postgres, then Alembic head vs the generated DDL with `drift.py` (expect `no drift`), which proves Plumbline's schema still loads without pgcrypto.

## Risks / Questions
- **Explorer not viewed in a browser.** The checks are a JavaScript syntax check and a grep. A mistake in the template's rendering code (for example a leftover reference to a removed variable) would only show when the page is opened. After this task, open `datamodel/models/plumbline/build/explorer.html` and tell me if any page errors.
- **The generated DDL changes** (the pgcrypto line is gone), so the drift job's input changes. The container run proves it still compares equal.
- **Type list slimmed** beyond the task text, per your "remove what Plumbline doesn't use" rule. It's reversible in one regex if you want some back.
- Docs still describe enums, immutable tables and `sample_rows` until 6.6.
