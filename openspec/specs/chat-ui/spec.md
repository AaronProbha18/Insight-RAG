# chat-ui Specification

## Purpose

What the chat interface shows and lets a user do: empty state, markdown answers, stop/copy/regenerate, scrolling, citations, upload, accessibility and theming. Introduced by the `assistant-ui-styled-chat` change (issue #3, ADR 0003).

## Requirements

### Requirement: Empty-state welcome and suggestions

The chat SHALL show a welcome message and at least one suggested question when the thread has no messages, and selecting a suggestion SHALL send it as the user's question.

#### Scenario: Empty thread shows welcome and suggestions

- **GIVEN** an empty thread
- **WHEN** the page loads
- **THEN** a welcome message and at least one suggested question are shown

#### Scenario: Selecting a suggestion sends it

- **GIVEN** an empty thread showing suggestions
- **WHEN** the user activates a suggestion by mouse or keyboard
- **THEN** that question is sent and the conversation replaces the welcome state

### Requirement: Markdown rendering of answers

Assistant answers SHALL render markdown (headings, lists, code blocks, tables) as formatted elements and SHALL NOT render raw HTML from the answer as HTML.

#### Scenario: Markdown is formatted

- **GIVEN** an assistant answer containing a list, a fenced code block and a table
- **WHEN** it renders
- **THEN** they display as list, code and table elements, not raw markdown text

#### Scenario: Raw HTML is not rendered

- **GIVEN** an assistant answer containing a `<script>` tag or other HTML
- **WHEN** it renders
- **THEN** no element is created from those tags

### Requirement: Stop a streaming answer

While an answer streams, the chat SHALL show a running indicator and a Stop control that halts streaming and keeps the partial answer.

#### Scenario: User stops streaming

- **GIVEN** an answer is streaming
- **WHEN** the user activates Stop
- **THEN** streaming stops and the partial answer stays visible

### Requirement: Message actions

Each completed assistant message SHALL offer Copy and Regenerate actions.

#### Scenario: Copy an answer

- **GIVEN** a completed assistant message
- **WHEN** the user activates Copy
- **THEN** the answer text is copied to the clipboard and the control confirms it

#### Scenario: Regenerate an answer

- **GIVEN** a completed assistant message
- **WHEN** the user activates Regenerate
- **THEN** the question is re-sent and the new answer replaces the old one

### Requirement: Scroll behaviour

The viewport SHALL follow streaming content while the user is at the bottom, SHALL NOT jump when the user has scrolled up, and SHALL then offer a scroll-to-bottom control.

#### Scenario: User scrolled up during streaming

- **GIVEN** the user has scrolled up in a long thread
- **WHEN** new content arrives
- **THEN** the view does not jump, and a scroll-to-bottom control is shown

### Requirement: Citations with answers

Each assistant answer with citations SHALL show its sources inside the message, each with title and locator, expandable to the quoted text and reachable by keyboard; an answer without citations SHALL show no sources section.

#### Scenario: Answer with citations

- **GIVEN** an assistant answer with citations in its metadata
- **WHEN** it renders
- **THEN** each citation shows its source title and locator and opens by keyboard to reveal the quoted text

#### Scenario: Abstention shows no sources

- **GIVEN** an assistant answer with an empty citation list
- **WHEN** it renders
- **THEN** no sources section is shown

### Requirement: Upload in the new kit

The upload control SHALL live on the Documents page, not above the chat,
and SHALL keep its behaviour: a labelled file input, and outcomes announced
in a polite live region.

#### Scenario: Upload outcome is announced

- **GIVEN** the person selects a file to upload on the Documents page
- **WHEN** the upload finishes with any outcome
- **THEN** the outcome text appears in a `role="status"` region

#### Scenario: Upload bar is gone

- **GIVEN** this change has landed
- **WHEN** the chat page shows
- **THEN** there is no upload bar or Documents dialog above the chat

### Requirement: Accessible, responsive, themed

The chat SHALL give every control an accessible name and a visible focus indicator, work by keyboard alone, have no horizontal scroll at 375px width, follow the OS light/dark preference, and meet WCAG AA (>= 4.5:1) for every text/background pair in both themes.

#### Scenario: Phone width in either theme

- **GIVEN** a 375px-wide viewport with the OS set to light or dark
- **WHEN** the user uses the chat
- **THEN** there is no horizontal scroll and every text/background pair meets 4.5:1

#### Scenario: Keyboard-only use

- **GIVEN** a keyboard-only user
- **WHEN** they upload a file, ask a question, stop, copy, regenerate and open a source
- **THEN** every step is reachable with a visible focus indicator

### Requirement: Chat with streamed answers

The frontend SHALL provide a chat built on assistant-ui that sends the
conversation to the backend chat endpoint and renders the answer as it
streams.

#### Scenario: Ask a question

- **WHEN** the user types a question and submits it
- **THEN** the answer appears incrementally in the thread

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

### Requirement: Feedback controls on answers

Each completed assistant message, abstentions included, SHALL offer
thumbs up and thumbs down; thumbs down SHALL offer an optional reason and
comment; the control SHALL show the recorded value. The controls SHALL be
keyboard-usable with accessible names, visible focus rings, and text
meeting 4.5:1 contrast in both themes.

#### Scenario: Keyboard only

- **GIVEN** the feedback controls
- **WHEN** used by keyboard only
- **THEN** they have accessible names and visible focus rings, and text meets 4.5:1 contrast in both themes

#### Scenario: No controls while streaming or on errors

- **GIVEN** an assistant message that is still running or ended with an error
- **WHEN** it is shown
- **THEN** no feedback controls are shown for it
