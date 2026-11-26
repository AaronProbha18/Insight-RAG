## Task
3.1 `models/plumbline/` with `model.yaml` and `documents`, `chunks`, `answers`, `feedback` YAML matching Alembic head (descriptions from `docs/architecture/overview.md`, synthetic sample rows only). Test: `make validate MODEL=models/plumbline` exits 0; ~~a test asserts the four tables and their column names equal the Alembic SQL's~~ (see Risks: owner decided to drop it and rely on the drift gate).

## Scope
- `datamodel/models/plumbline/model.yaml`:
  ```yaml
  id_type: bigserial
  public_id_check: false
  updated_at_trigger: false
  constraint_names: postgres
  comments: false
  ```
- `domains/rag/domain.yaml`: domain `rag`, layer `runtime`, `load_order: 10`, no dependencies.
- Four tables under `domains/rag/tables/`. Each one is checked against `backend/migrations/versions/0001_initial.py` and `0002_answers_feedback.py` (read-only, only while writing the YAML).

  | Table | `system_columns` | `public_id_prefix` | Columns written in YAML |
  | --- | --- | --- | --- |
  | `documents` | `[public_id, created_at]` | `doc` | `content_hash char(64)` (unique), `title`, `filename`, `connector` (all `text`) |
  | `chunks` | `[]` | none | `document_id bigint` → `documents.id`, `on_delete: cascade`; `ordinal int`; `text`; `locator`; `embedding vector({embedding_dim})`; `unique: [[document_id, ordinal]]` |
  | `answers` | `[public_id, created_at]` | `ans` | `question`, `answer` (`text`), `abstained bool`, `citations jsonb`, `chat_model`, `embedding_model` (`text`), `abstain_threshold double precision` |
  | `feedback` | `[updated_at]` | none | `answer_id bigint` → `answers.id`, `on_delete: cascade`, unique; `rating text`; `reason text` (nullable); `comment text` (nullable); table checks (below) |

- `feedback` checks are written as table `checks`, not `enums/`: `validate.py` requires UPPER_SNAKE enum values, and Plumbline's are lowercase (`up`, `wrong_source`). Four checks: `rating IN ('up', 'down')`; `reason IS NULL OR reason IN ('wrong', 'incomplete', 'wrong_source', 'should_have_answered', 'other')`; `comment IS NULL OR char_length(comment) <= 2000`; `reason IS NULL OR rating = 'down'`. In `postgres` naming mode the written names are kept.
- Table and column `description`s are taken from the "Data" table in `docs/architecture/overview.md` (lines 158 to 161), reworded only where that table is silent. `status: published` on all four.
- `sample_rows`: two synthetic rows each for `documents`, `answers`, `feedback`. Titles, questions and answers are obviously fake (`Example handbook`, `What is the example policy?`). `chunks` gets none, since its embedding is a vector. No real data.
- Guides read: `docs/guides/backend-development-guide.md` is not needed (nothing under `backend/` changes). I read overview.md lines 150 to 165 and both Alembic revisions.

## Not in scope
- Any test comparing YAML with Alembic (decided: the drift gate in 4.1 and 4.2 does it, against real databases).
- `drift.py`, the CI job and docs (4.x and 5.1).
- Changing Alembic or the backend.
- A `migrations/` folder or `model.lock.json` for this model: it mirrors Alembic, so `make migrate` and `make check` don't apply to it.

## Files
| File | Change |
| --- | --- |
| `datamodel/models/plumbline/model.yaml` | new |
| `datamodel/models/plumbline/domains/rag/domain.yaml` | new |
| `datamodel/models/plumbline/domains/rag/tables/{documents,chunks,answers,feedback}.yaml` | new |
| `openspec/changes/datamodel-plumbline-mirror/tasks.md` | reword the 3.1 test line to drop the Alembic-comparison test (decision above) |
| `datamodel/tests/test_plumbline_model.py` | new: runs validate on the model, and checks the generated DDL's four table names and column order against the lists in this plan |

## Tests
- `uv run python tools/validate.py --model models/plumbline` exits 0 and reports 1 domain, 4 tables, 0 enums, 0 errors (spec: *Model valid*).
- `test_plumbline_model.py`:
  - With `PLUMBLINE_EMBEDDING_DIM=8`, `build_objects(load(model_dir("models/plumbline")))` has exactly `documents`, `chunks`, `answers`, `feedback`.
  - Each table's column list equals the one in this plan: `id, public_id, content_hash, title, filename, connector, created_at` for `documents`, and likewise for the other three. The plan's table is where the "mirrors Alembic" claim is recorded until the drift gate lands.
  - `id` is `bigserial`, `chunks.embedding` is `vector(8)`, and there is no `public_id` on `chunks` or `feedback`.
- The sample's suite and DDL are unchanged.

## Checks to run
- `cd datamodel && uv run python tools/validate.py --model models/plumbline`
- `PLUMBLINE_EMBEDDING_DIM=8 uv run python tools/ddl.py --model models/plumbline` (eyeball the SQL against Alembic once; a real comparison waits for 4.1)
- `uv run python -m pytest -q -rs tests`
- `uv run python tools/validate.py && uv run python tools/migrate.py --dry-run` (sample: "no changes")

## Risks / Questions
- `question`, `answer`, `comment` are free text a member could type PHI into. They're set to `pii: none` here because YAML `pii` only drives the explorer today, but that probably understates it. OPEN QUESTION for the business tech review / CMO: should these be `indirect`? I've added it as an `open_questions` entry on `answers` (owner `CMO`) and set that table to `status: draft`, because `validate.py` rejects a published table with an open CMO question. So `answers` is `draft`, the other three `published`.
- Whether the YAML check expressions render the same as Alembic's inline CHECKs through `pg_get_constraintdef` is only knowable on a real Postgres. UNVERIFIED until the drift job runs (4.2); names and comments are ignored by the gate anyway.
- The column order matches Alembic's (`id`, `public_id`, body, `created_at`/`updated_at`), but I don't yet know whether the gate compares order. 4.1 decides that.
- tasks.md and the spec describe a column-name test against Alembic. Dropping it is the owner's call from this session; the spec scenario *Model valid* is still covered.
