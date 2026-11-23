## ADDED Requirements

### Requirement: Product exposes the Plumbline names

The system SHALL present itself as Plumbline in the UI title, README,
docs, Docker service and image names, and SHALL use `PLUMBLINE_*` env
vars and the `plumbline-eval` command in every documented step.

#### Scenario: Fresh clone quickstart

- **GIVEN** a fresh clone
- **WHEN** someone follows the README quickstart
- **THEN** the app starts as Plumbline, and every documented command and variable uses the new names

### Requirement: Legacy env vars stop startup

The system MUST NOT read `INSIGHT_RAG_*` variables. If any is set at
startup, it SHALL stop with a message naming the `PLUMBLINE_*`
replacement for each one.

#### Scenario: Old env file

- **GIVEN** a `.env` that still sets `INSIGHT_RAG_*` variables
- **WHEN** the app starts
- **THEN** it stops with a message naming each new variable

### Requirement: Legacy connector group is skipped with a warning

The registry SHALL load connectors only from `plumbline.connectors`. For
each entry under `insight_rag.connectors` it SHALL log a warning naming
the new group and SHALL NOT load it.

#### Scenario: Third-party connector on old group

- **GIVEN** a third-party connector registered under `insight_rag.connectors`
- **WHEN** the registry loads connectors
- **THEN** it is skipped with a warning that names `plumbline.connectors`

### Requirement: Existing databases have an upgrade path

The migration notes SHALL state exactly how to keep documents indexed in
a database created with the old default names.

#### Scenario: Upgrade with old database

- **GIVEN** a local database created with the old default names
- **WHEN** the user upgrades and follows the migration note
- **THEN** their indexed documents are still available

### Requirement: Rename does not change answers

Built-in connectors SHALL pass conformance, and eval metrics SHALL equal
the pre-rename baseline.

#### Scenario: Conformance and eval after rename

- **GIVEN** the built-in PDF and DOCX connectors
- **WHEN** conformance and `plumbline-eval` run
- **THEN** conformance passes and every metric matches the pre-rename baseline

### Requirement: Old names only where allowed

Outside ADR 0004, the migration notes, the legacy-name checks, and
`openspec/changes/**`, the repository SHALL contain no `insight-rag`,
`insight_rag` or `INSIGHT_RAG`.

#### Scenario: Repository search

- **GIVEN** the repository
- **WHEN** searched for `insight-rag`, `insight_rag` or `INSIGHT_RAG`
- **THEN** matches appear only in ADR 0004, the migration notes, the legacy-name checks and `openspec/changes/**`

### Requirement: Distribution name

The backend distribution SHALL be named `plumbline-rag`.

#### Scenario: Build

- **GIVEN** the backend package
- **WHEN** built
- **THEN** the distribution name is `plumbline-rag`
