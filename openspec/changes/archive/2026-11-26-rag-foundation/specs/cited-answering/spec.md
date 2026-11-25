## ADDED Requirements

### Requirement: Grounded answer with citations

The system SHALL answer a question using only retrieved chunks and SHALL
return, with every answer, citations identifying the document title and
locator of each chunk the answer relies on.

#### Scenario: Answerable question

- **GIVEN** an indexed document states the refund window is 30 days
- **WHEN** the user asks "What is the refund window?"
- **THEN** the streamed answer states 30 days
- **AND** the citations include that document's title and the locator of the supporting chunk

### Requirement: Abstain when the sources do not answer

The system SHALL abstain - stating the answer is not in the indexed
sources - when retrieval finds no chunk above the relevance threshold, or
when the retrieved chunks do not support an answer. An abstention SHALL
carry no citations.

#### Scenario: Question outside the corpus

- **GIVEN** no indexed document mentions shipping times
- **WHEN** the user asks "How long does shipping take?"
- **THEN** the answer says this is not covered by the indexed documents and includes no citations

#### Scenario: Empty index

- **GIVEN** no documents are indexed
- **WHEN** the user asks any question
- **THEN** the system abstains without calling the LLM for generation

### Requirement: Streamed response

The system SHALL stream the answer incrementally and SHALL deliver
citations and an end-of-response marker in the same stream.

#### Scenario: Stream completes

- **WHEN** a question is sent to the chat endpoint
- **THEN** the client receives text deltas, then the citations, then an end marker

### Requirement: No self-reported confidence

The system SHALL NOT display or return an LLM-generated confidence
percentage for an answer.

#### Scenario: Answer payload

- **WHEN** any answer is returned
- **THEN** it contains no confidence score field
