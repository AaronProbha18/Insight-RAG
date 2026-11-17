## ADDED Requirements

### Requirement: Chat with streamed answers

The frontend SHALL provide a chat built on assistant-ui that sends the
conversation to the backend chat endpoint and renders the answer as it
streams.

#### Scenario: Ask a question

- **WHEN** the user types a question and submits it
- **THEN** the answer appears incrementally in the thread

### Requirement: Citations are visible

The frontend SHALL show each citation of an answer with its document title
and locator, and SHALL show the cited chunk text on demand.

#### Scenario: Inspect a citation

- **GIVEN** an answer with two citations
- **WHEN** the user opens a citation
- **THEN** the document title, locator, and cited text are shown

#### Scenario: Abstention shows no sources

- **WHEN** the backend abstains
- **THEN** the message is shown with no citation list

### Requirement: Upload documents from the UI

The frontend SHALL let the user upload PDF and DOCX files and SHALL show
the outcome: indexed, already indexed, unsupported type, or unreadable
file.

#### Scenario: Duplicate upload feedback

- **GIVEN** a file was already indexed
- **WHEN** the user uploads it again
- **THEN** the UI says the document is already indexed

#### Scenario: Unsupported file feedback

- **WHEN** the user uploads a `.png`
- **THEN** the UI shows an error listing the supported types

### Requirement: Accessible basics

The chat input, send control, upload control, and citation toggles SHALL
be operable by keyboard and have accessible names.

#### Scenario: Keyboard-only use

- **WHEN** a user navigates with the keyboard only
- **THEN** they can upload a file, ask a question, and open a citation
