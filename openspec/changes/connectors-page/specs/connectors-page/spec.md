## ADDED Requirements

### Requirement: List loaded connectors

The system SHALL provide `GET /connectors`, returning each loaded connector's
name, file extensions and contract version, and nothing else. Connectors
that failed to load or declare a different contract version SHALL NOT be
listed.

#### Scenario: Built-in connectors

- **GIVEN** the built-in PDF and Word connectors
- **WHEN** `GET /connectors` is called
- **THEN** it returns both with their name, extensions and contract version, and no internal paths or ids

#### Scenario: Failed third-party connector

- **GIVEN** a third-party connector that failed to load
- **WHEN** `GET /connectors` is called
- **THEN** it is not listed

### Requirement: Connectors page

The app SHALL show a Connectors page at `/app/connectors` with one card per
listed connector, showing its name, extensions, the number of indexed
documents that use it, its latest quality result, and when and with which
models it was last checked.

#### Scenario: Cards for each connector

- **GIVEN** the PDF and Word connectors and three indexed PDFs
- **WHEN** the Connectors page opens
- **THEN** there is a card for each, and the PDF card shows its extensions and 3 documents
