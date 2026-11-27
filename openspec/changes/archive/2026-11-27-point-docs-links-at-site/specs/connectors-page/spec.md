## MODIFIED Requirements

### Requirement: Connectors page

The app SHALL show a Connectors page at `/app/connectors`, titled
"Connectors and quality", with an intro sentence, a line explaining the 0.80
bar with a "How it's measured" link, and one card per listed connector. Each
card SHALL show the connector's name, "Built-in" when it is, its extensions,
the number of indexed documents that use it, and its latest quality result.
The page SHALL end with a "Build your own" card linking to the connector
guide, and SHALL NOT show planned or not-yet-built connectors. The "How it's
measured" and connector guide links SHALL open pages on the deployed docs
site, not Markdown files on GitHub.

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

#### Scenario: Connector guide opens on the docs site

- **GIVEN** the Connectors page
- **WHEN** the person clicks Connector guide
- **THEN** the deployed connector development guide opens (`/guides/connector-development-guide/`)

#### Scenario: How it's measured opens on the docs site

- **GIVEN** the Connectors page
- **WHEN** the person clicks How it's measured
- **THEN** the deployed architecture overview opens at its evaluation section (`/architecture/overview/#evaluate-plumbline-eval`)
