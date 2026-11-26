# product-naming Specification

## Purpose

The names Plumbline presents to users, operators and connector authors; how the pre-rename names are rejected; the upgrade path for existing databases; and the brand assets that ship with it. Introduced by the `rename-to-plumbline` change (issue #11).

## Requirements

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

The README SHALL state exactly how to keep documents indexed in
a database created with the old default names.

#### Scenario: Upgrade with old database

- **GIVEN** a local database created with the old default names
- **WHEN** the user upgrades and follows the README note
- **THEN** their indexed documents are still available

### Requirement: Rename does not change answers

Built-in connectors SHALL pass conformance, and eval metrics SHALL equal
the pre-rename baseline.

#### Scenario: Conformance and eval after rename

- **GIVEN** the built-in PDF and DOCX connectors
- **WHEN** conformance and `plumbline-eval` run
- **THEN** conformance passes and every metric matches the pre-rename baseline

### Requirement: Old names only where allowed

Outside the legacy-name checks, their tests, this spec, and
`openspec/changes/**`, the repository SHALL contain no `insight-rag`,
`insight_rag` or `INSIGHT_RAG`.

#### Scenario: Repository search

- **GIVEN** the repository
- **WHEN** searched for `insight-rag`, `insight_rag` or `INSIGHT_RAG`
- **THEN** matches appear only in the legacy-name checks, their tests, this spec and `openspec/changes/**`

### Requirement: Distribution name

The backend distribution SHALL be named `plumbline-rag`.

#### Scenario: Build

- **GIVEN** the backend package
- **WHEN** built
- **THEN** the distribution name is `plumbline-rag`

### Requirement: Brand assets ship with the product

The repository SHALL keep the supplied Plumbline brand files under
`docs/brand/`, the web app SHALL serve its favicon and Apple touch icon,
and the README SHALL show the Plumbline logo, switching with the reader's
light or dark theme.

#### Scenario: Built UI carries the icons

- **GIVEN** a built frontend
- **WHEN** `index.html` is loaded
- **THEN** each icon it links (`favicon.svg`, `favicon.ico`, `apple-touch-icon.png`) exists in the build output

#### Scenario: README logo resolves

- **GIVEN** the README header
- **WHEN** its light and dark logo links are followed
- **THEN** both files exist under `docs/brand/`

