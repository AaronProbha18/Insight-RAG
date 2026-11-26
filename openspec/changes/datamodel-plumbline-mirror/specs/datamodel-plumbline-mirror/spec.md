## ADDED Requirements

### Requirement: Model selection

The datamodel tools SHALL operate on the model folder given by `--model`
(Makefile `MODEL`), defaulting to `models/sample`.

#### Scenario: Default model unchanged

- **GIVEN** no `--model`
- **WHEN** `make check test` runs
- **THEN** it validates and tests the sample model, as before the move

#### Scenario: Named model

- **GIVEN** `MODEL=models/plumbline`
- **WHEN** `make validate ddl explorer` runs
- **THEN** it reads that folder's YAML and writes that folder's `build/`

### Requirement: Per-model conventions

A model's optional `model.yaml` SHALL set `id_type`, `public_id_check`,
`updated_at_trigger`, `constraint_names` and `comments`; with no
`model.yaml`, the generated DDL SHALL be identical to before this change.

#### Scenario: No model.yaml keeps today's output

- **GIVEN** the sample model with no `model.yaml`
- **WHEN** DDL is generated
- **THEN** it equals the sample's committed lock (no unmigrated changes)

#### Scenario: bigserial keys

- **GIVEN** `id_type: bigserial`
- **WHEN** DDL is generated
- **THEN** `id` is `bigserial` and foreign key columns must be `bigint`; a `uuid` foreign key fails validation

#### Scenario: Conventions switched off

- **GIVEN** `public_id_check: false`, `updated_at_trigger: false`, `comments: false`, `constraint_names: postgres`
- **WHEN** DDL is generated
- **THEN** there is no `public_id` CHECK, no `updated_at` trigger or `set_updated_at()` function, no `COMMENT ON`, and constraints use Postgres default names

### Requirement: Per-table system columns

A table's `system_columns` SHALL limit the injected optional system
columns to the listed subset of `public_id`, `created_at` and `updated_at`;
`id` SHALL always be injected.

#### Scenario: Subset injected

- **GIVEN** a table with `system_columns: [created_at]`
- **WHEN** DDL is generated
- **THEN** it has `id` and `created_at`, and no `public_id` or `updated_at`

#### Scenario: Unknown entry

- **GIVEN** `system_columns: [id]` or an unknown name
- **WHEN** validating
- **THEN** it fails naming the table

### Requirement: Extra column types

The model SHALL accept `char(n)`, `double precision`, `vector(n)` and
`vector({embedding_dim})`; generation SHALL fill `{embedding_dim}` from
`PLUMBLINE_EMBEDDING_DIM`.

#### Scenario: Embedding dimension filled

- **GIVEN** a `vector({embedding_dim})` column and `PLUMBLINE_EMBEDDING_DIM=8`
- **WHEN** DDL is generated
- **THEN** the column is `vector(8)`

#### Scenario: Dimension missing

- **GIVEN** a `vector({embedding_dim})` column and `PLUMBLINE_EMBEDDING_DIM` unset
- **WHEN** DDL is generated
- **THEN** it fails with a message naming `PLUMBLINE_EMBEDDING_DIM`

### Requirement: Plumbline model mirrors Alembic

`models/plumbline/` SHALL describe every table and column of Plumbline's
Alembic head, and SHALL pass validation.

#### Scenario: Model valid

- **GIVEN** `models/plumbline`
- **WHEN** `make validate MODEL=models/plumbline` runs
- **THEN** it exits 0

### Requirement: Drift gate

CI SHALL fail when the schema generated from `models/plumbline` differs
from Alembic head in columns, types, nullability, defaults, constraint
definitions, indexes or triggers; it SHALL ignore object names, comments
and `alembic_version`.

#### Scenario: In sync

- **GIVEN** Alembic head and `models/plumbline` describe the same schema
- **WHEN** the drift job runs
- **THEN** `tools/drift.py` exits 0

#### Scenario: Alembic changed without YAML

- **GIVEN** a new column in an Alembic revision and no matching YAML
- **WHEN** the drift job runs
- **THEN** it exits 1, naming the table and column

#### Scenario: YAML changed without Alembic

- **GIVEN** a YAML-only change (e.g. a column made nullable)
- **WHEN** the drift job runs
- **THEN** it exits 1, naming the table and column

#### Scenario: Names and comments ignored

- **GIVEN** the same structure with different constraint names and comments
- **WHEN** `tools/drift.py` compares them
- **THEN** it exits 0
