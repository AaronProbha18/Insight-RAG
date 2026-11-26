## ADDED Requirements

### Requirement: Model validation

`make validate` SHALL exit 0 for a valid model and SHALL fail, naming the
offending item, for each rule below.

#### Scenario: Valid model

- **GIVEN** a valid model
- **WHEN** `make validate` runs
- **THEN** it exits 0

#### Scenario: Foreign key outside depends_on, not at id, or not bigint

- **GIVEN** a FK to a table outside the domain's `depends_on`, not pointing at `id`, or on a column that is not `bigint`
- **WHEN** `make validate` runs
- **THEN** it fails naming the column

#### Scenario: Identifier too long

- **GIVEN** an identifier longer than 63 bytes
- **WHEN** `make validate` runs
- **THEN** it fails (no silent truncation)

#### Scenario: Duplicate prefix or question id

- **GIVEN** a duplicate `public_id_prefix`, or a duplicate `OQ-<TABLE>-` id
- **WHEN** validating
- **THEN** it fails

#### Scenario: Default on a direct identifier or PHI column

- **GIVEN** a `pii: direct` or `phi` column with a `default`
- **WHEN** validating
- **THEN** it fails

#### Scenario: Published with an open CMO question

- **GIVEN** a `published` table with an open question owned by the CMO
- **WHEN** validating
- **THEN** it fails

#### Scenario: Answered question without an answer

- **GIVEN** a question marked answered/closed with no `answer`
- **WHEN** validating
- **THEN** it fails

### Requirement: Generated DDL

Generated DDL SHALL inject `id bigserial` and the optional system columns a
table lists in `system_columns`, and SHALL use Postgres's own constraint
names, with no comments, no `public_id` CHECK and no triggers.

#### Scenario: System columns injected

- **GIVEN** a table YAML
- **WHEN** DDL is generated
- **THEN** `id bigserial` is the primary key, and `public_id` (unique), `created_at` and `updated_at` appear only if the table lists them in `system_columns`; none is written in YAML

#### Scenario: Postgres-style names and no extras

- **GIVEN** a model with a foreign key, a unique column and a check
- **WHEN** DDL is generated
- **THEN** constraints are named `<t>_pkey`, `<t>_<c>_fkey` and `<t>_<cols>_key` (a YAML check keeps its written name), and there is no `COMMENT ON`, `public_id` CHECK or trigger

## REMOVED Requirements

### Requirement: Validator gates the model

**Reason**: Replaced by "Model validation" in this change: the same gate, without the sample-row, business-rule and enum rules, and with foreign keys required to be `bigint`.

**Migration**: Use `make validate`; the remaining rules are unchanged.

### Requirement: DDL generation

**Reason**: Replaced by "Generated DDL" in this change: enums, immutable tables and uuid ids are gone, and the DDL follows Plumbline's conventions.

**Migration**: Use `make ddl`; `id` is `bigserial` and constraint names are Postgres's own.

### Requirement: Migrations

**Reason**: Plumbline's migrations are Alembic's, hand-written in `backend/migrations/`. The toolkit no longer generates migrations, a lock file or destructive-change approvals.

**Migration**: Write an Alembic revision, then update `datamodel/models/plumbline/` in the same PR; the `plumbline-drift` job checks the two agree.

### Requirement: CI gate

**Reason**: The gate compared the YAML with a migration lock file. It is replaced by the `plumbline-drift` job, specified in `datamodel-plumbline-mirror`, which compares the YAML's generated schema with Alembic head.

**Migration**: The `datamodel` job now runs `make validate test`; the drift comparison runs in `plumbline-drift`.
