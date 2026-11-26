# datamodel-plumbline-mirror Specification

## Purpose
The toolkit in `datamodel/` mirrors Plumbline's Alembic schema in YAML and fails CI when the two disagree. This capability covers model selection, Plumbline's conventions, the Plumbline model, the drift gate, and the limits on what the toolkit supports.

## Requirements

### Requirement: Model selection

The datamodel tools SHALL operate on the model folder given by `--model`
(Makefile `MODEL`), defaulting to `models/plumbline`.

#### Scenario: Default is Plumbline's model

- **GIVEN** no `--model`
- **WHEN** `make validate ddl` runs
- **THEN** it reads `models/plumbline` and writes `models/plumbline/build/`

#### Scenario: Named model

- **GIVEN** `--model <dir>` for another model folder
- **WHEN** a tool runs
- **THEN** it reads that folder's YAML and writes that folder's `build/`

### Requirement: Plumbline conventions

The generated DDL SHALL follow Plumbline's conventions with no per-model
switches: `id bigserial` with `bigint` foreign keys, no `public_id` CHECK,
no `updated_at` trigger, no `COMMENT ON`, and Postgres-style constraint
names (`<t>_pkey`, `<t>_<c>_fkey`, `<t>_<cols>_key`).

#### Scenario: Plumbline conventions generated

- **GIVEN** `models/plumbline`
- **WHEN** DDL is generated
- **THEN** `id` is `bigserial`, foreign keys are `bigint`, there is no `public_id` CHECK, no `updated_at` trigger or `set_updated_at()` function, no `COMMENT ON`, and constraints use Postgres default names

#### Scenario: uuid foreign key rejected

- **GIVEN** a foreign key column typed `uuid`
- **WHEN** validating
- **THEN** it fails, naming the table, column and the expected type `bigint`

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
Alembic head, SHALL contain no table or column that Alembic head lacks, and
SHALL pass validation.

#### Scenario: Model valid

- **GIVEN** `models/plumbline`
- **WHEN** `make validate` runs
- **THEN** it exits 0

### Requirement: Drift gate

CI SHALL fail when the schema generated from `models/plumbline` differs
from Alembic head in columns, types, nullability, defaults, constraint
definitions, indexes or triggers; it SHALL ignore object names, comments,
column order and `alembic_version`.

#### Scenario: In sync

- **GIVEN** Alembic head and `models/plumbline` describe the same schema
- **WHEN** the drift job runs
- **THEN** `tools/drift.py` exits 0

#### Scenario: Alembic changed without YAML

- **GIVEN** a new column in an Alembic revision and no matching YAML
- **WHEN** the drift job runs
- **THEN** it exits 1, naming the table and column

#### Scenario: YAML changed without Alembic

- **GIVEN** a YAML-only change (e.g. a column made nullable, or a column Alembic lacks)
- **WHEN** the drift job runs
- **THEN** it exits 1, naming the table and column

#### Scenario: Names and comments ignored

- **GIVEN** the same structure with different constraint names and comments
- **WHEN** `tools/drift.py` compares them
- **THEN** it exits 0

#### Scenario: Output names the sides

- **GIVEN** a difference between the two databases
- **WHEN** the drift job prints it
- **THEN** each line says whether the item is only in `alembic` or only in `yaml`

### Requirement: Toolkit limited to what Plumbline uses

The datamodel toolkit SHALL NOT support migration generation, enums,
immutable tables, uuid primary keys, `sample_rows`, `business_rules` or
`model.yaml` settings.

#### Scenario: Removed field rejected

- **GIVEN** a table YAML that sets `immutable`, `sample_rows` or `business_rules`, or a column that sets `enum`
- **WHEN** the model loads
- **THEN** it fails, naming the field

#### Scenario: Migration tooling absent

- **GIVEN** the `datamodel/` folder
- **WHEN** listing its tools and Makefile targets
- **THEN** there is no `migrate.py`, `model.lock.json`, `make migrate` or `make check`
