# connector-contract Specification

## Purpose

The protocol, discovery and conformance kit for ingestion connectors. Introduced by the `rag-foundation` change.

## Requirements

### Requirement: Connector discovery via entry points

The system SHALL load ingestion connectors from the `insight_rag.connectors`
Python entry-point group, including built-in connectors, so that an
installed third-party package is available without changes to core.

#### Scenario: Third-party connector is discovered

- **GIVEN** a package registering a connector under `insight_rag.connectors` is installed
- **WHEN** the backend starts
- **THEN** the connector appears in the registry and its file types are accepted for upload

#### Scenario: Built-in connectors load the same way

- **WHEN** the backend starts with no extra packages installed
- **THEN** the `pdf` and `docx` connectors are registered through entry points

### Requirement: Contract version check

The system SHALL refuse to register a connector whose `contract_version`
differs from the supported contract version, and SHALL log the
connector name and both versions.

#### Scenario: Incompatible connector is skipped

- **GIVEN** an installed connector declares `contract_version` 2 while core supports 1
- **WHEN** the backend starts
- **THEN** the connector is not registered, a warning names it and both versions, and startup continues

### Requirement: Conformance kit

The system SHALL provide an importable conformance test function that any
connector package can run against its own fixture files, failing when the
connector violates the contract.

#### Scenario: Conforming connector passes

- **GIVEN** a connector and a directory of valid fixture files
- **WHEN** the conformance kit runs
- **THEN** it passes: versions match, every parsed document has at least one section, every section has a non-empty locator, and parsing identical bytes twice yields identical output

#### Scenario: Corrupt input is rejected cleanly

- **GIVEN** a corrupt file of a type the connector claims
- **WHEN** the conformance kit runs it through the connector
- **THEN** the connector raises the contract's parse error type rather than crashing with an unrelated exception
