# connectors-page Specification

## Purpose

Which connectors Plumbline has loaded and how each one measures up: the read-only `GET /connectors` listing and the Connectors and quality page at `/app/connectors`, with one card per connector. Introduced by the `connectors-page` change (issue #34).

## Requirements

### Requirement: List loaded connectors

The system SHALL provide `GET /connectors`, returning each loaded connector's
name, file extensions, contract version and whether it is built in
(`builtin`), and nothing else. Connectors
that failed to load or declare a different contract version SHALL NOT be
listed.

#### Scenario: Built-in connectors

- **GIVEN** the built-in PDF and Word connectors
- **WHEN** `GET /connectors` is called
- **THEN** it returns both with their name, extensions, contract version and `builtin` true, and no internal paths or ids

#### Scenario: Third-party connector

- **GIVEN** a loaded connector that is not part of Plumbline
- **WHEN** `GET /connectors` is called
- **THEN** its `builtin` is false

#### Scenario: Failed third-party connector

- **GIVEN** a third-party connector that failed to load
- **WHEN** `GET /connectors` is called
- **THEN** it is not listed

### Requirement: Connectors page

The app SHALL show a Connectors page at `/app/connectors`, titled
"Connectors and quality", with an intro sentence, a line explaining the 0.80
bar with a "How it's measured" link, and one card per listed connector. Each
card SHALL show the connector's name, "Built-in" when it is, its extensions,
the number of indexed documents that use it, and its latest quality result.
The page SHALL end with a "Build your own" card linking to the connector
guide, and SHALL NOT show planned or not-yet-built connectors.

#### Scenario: Cards for each connector

- **GIVEN** the PDF and Word connectors and three indexed PDFs
- **WHEN** the Connectors page opens
- **THEN** there is a card for each, and the PDF card shows its extensions and 3 documents

#### Scenario: Page frame

- **GIVEN** the Connectors page
- **WHEN** it opens
- **THEN** it shows the title "Connectors and quality", the intro, the 0.80 explainer with its link, and the "Build your own" card, and no card for a connector that is not loaded

#### Scenario: Subtitle

- **GIVEN** a built-in PDF connector and three indexed PDFs
- **WHEN** the Connectors page opens
- **THEN** its card reads "Built-in", ".pdf" and "3 documents"
