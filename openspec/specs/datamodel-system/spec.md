# datamodel-system Specification

## Purpose
TBD - created by archiving change datamodel-system. Update Purpose after archive.
## Requirements

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

### Requirement: Explorer

`make explorer` SHALL produce one self-contained HTML page that renders
offline from a local file and escapes YAML text. It is not hosted or
uploaded.

#### Scenario: Offline render

- **GIVEN** `make explorer`
- **WHEN** `build/explorer.html` opens offline
- **THEN** it renders the model with no console errors and YAML text is escaped
