## ADDED Requirements

### Requirement: Unanswered list endpoint

`GET /unanswered` SHALL return refused answers and answers rated thumbs down, grouped by question after ignoring case and extra spaces. Each group SHALL carry the latest question text, the count and last date of in-range events, and the latest event's reason. It SHALL accept `days` of 7, 30 or 90 (omitted means all time), SHALL return 422 for any other value, and SHALL NOT include retrieval scores, confidence or answer text.

#### Scenario: Same question grouped

- **GIVEN** the same question refused three times, differing only in case and spacing
- **WHEN** the Unanswered page opens
- **THEN** it shows one row, "Asked 3 times", with the latest date

#### Scenario: Marked wrong with a reason

- **GIVEN** an answer marked wrong with the reason "wrong source"
- **WHEN** the page opens
- **THEN** its row shows "Marked wrong: wrong source"

#### Scenario: Outside the range

- **GIVEN** a question last asked 40 days ago
- **WHEN** the range is Last 30 days
- **THEN** it is not listed

#### Scenario: Invalid range

- **GIVEN** a request with `days=5`
- **WHEN** it is sent
- **THEN** the API returns 422 with no internal details

### Requirement: Unanswered page

An Unanswered page SHALL show one row per group with the question, "Asked N times", the last date, the reason as text and an Add a document link. It SHALL offer a range filter of Last 7, 30 and 90 days and All time, defaulting to Last 30 days, and an empty state when nothing is listed.

#### Scenario: Empty state

- **GIVEN** no refused or wrong answers
- **WHEN** the page opens
- **THEN** an empty state explains what will appear here

#### Scenario: Add a document

- **GIVEN** a row
- **WHEN** the person clicks Add a document
- **THEN** the Documents page opens

#### Scenario: Keyboard, themes and contrast

- **GIVEN** the page
- **WHEN** used by keyboard, in light and dark mode
- **THEN** every control is reachable, reasons are text and not colour alone, and text meets 4.5:1 contrast

### Requirement: Entry points from the sidebar and the chat

The sidebar SHALL show an Unanswered item with the number of groups in the last 30 days, refreshed after each chat answer. A refused answer in the chat SHALL link to the Unanswered page with the text "Added to Unanswered".

#### Scenario: Refusal links and the count updates

- **GIVEN** a refused question
- **WHEN** the answer appears in the chat
- **THEN** the chat links to the Unanswered page and the sidebar count updates
