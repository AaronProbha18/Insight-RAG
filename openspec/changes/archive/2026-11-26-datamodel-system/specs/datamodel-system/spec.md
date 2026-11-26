## ADDED Requirements

### Requirement: Validator gates the model

`make validate` SHALL exit 0 for a valid model and SHALL fail, naming the
offending item, for each rule below.

#### Scenario: Valid model

- **GIVEN** a valid model
- **WHEN** `make validate` runs
- **THEN** it exits 0

#### Scenario: Foreign key outside depends_on or not at id

- **GIVEN** a FK to a table outside the domain's `depends_on`, or not pointing at `id`
- **WHEN** `make validate` runs
- **THEN** it fails naming the column

#### Scenario: Identifier too long

- **GIVEN** an identifier longer than 63 bytes
- **WHEN** `make validate` runs
- **THEN** it fails (no silent truncation)

#### Scenario: Duplicate prefix or rule/question id

- **GIVEN** a duplicate `public_id_prefix`, or a duplicate `BR-<TABLE>-` / `OQ-<TABLE>-` id
- **WHEN** validating
- **THEN** it fails

#### Scenario: Default on a direct identifier or PHI column

- **GIVEN** a `pii: direct` or `phi` column with a `default`
- **WHEN** validating
- **THEN** it fails

#### Scenario: Bad sample row

- **GIVEN** a sample row with an unknown column, invalid enum value, SSN-shaped string or non-reserved email domain
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

### Requirement: DDL generation

Generated DDL SHALL inject the system columns and render enums as CHECK
constraints and immutable tables with a guarding trigger.

#### Scenario: System columns injected

- **GIVEN** a table YAML
- **WHEN** DDL is generated
- **THEN** `id` (uuid PK), `public_id` (unique, `^prefix_[0-9A-Za-z]+$`), `created_at`, and `updated_at` (unless fully immutable) are injected and never written in YAML

#### Scenario: Immutable table trigger

- **GIVEN** an immutable table
- **WHEN** a DELETE or an UPDATE outside `mutable_columns` runs
- **THEN** the generated trigger rejects it; an allow-listed UPDATE succeeds and advances `updated_at`

#### Scenario: Enum as CHECK

- **GIVEN** an enum column
- **WHEN** DDL is generated
- **THEN** it is a CHECK constraint, not a Postgres ENUM type

### Requirement: Migrations

`make migrate` SHALL write the next numbered migration and the updated
`model.lock.json`, refuse destructive changes without
`--allow-destructive`, and produce a schema equivalent to a fresh build.

#### Scenario: Migration written

- **GIVEN** a YAML change
- **WHEN** `make migrate NAME=x` runs
- **THEN** `migrations/NNNN_x.sql` and an updated `model.lock.json` are written

#### Scenario: Destructive change

- **GIVEN** a destructive change (drop, type change, NOT NULL without default)
- **WHEN** migrating without `--allow-destructive`
- **THEN** it is refused; with the flag the file records the approval

#### Scenario: Equivalent to a fresh build

- **GIVEN** baseline plus incremental migrations applied on PostgreSQL 16
- **WHEN** compared via `pg_dump -s` with a fresh `build/schema.sql`
- **THEN** the schemas are equivalent (column order excepted)

### Requirement: CI gate

CI SHALL fail when the YAML changed without a matching migration.

#### Scenario: Unmigrated change

- **GIVEN** the YAML moved without a migration
- **WHEN** CI runs `make check test`
- **THEN** it fails

### Requirement: Explorer

`make explorer` SHALL produce one self-contained HTML page that renders
offline from a local file and escapes YAML text. It is not hosted or
uploaded.

#### Scenario: Offline render

- **GIVEN** `make explorer`
- **WHEN** `build/explorer.html` opens offline
- **THEN** it renders the model with no console errors and YAML text is escaped
