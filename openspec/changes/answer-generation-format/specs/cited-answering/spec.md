## ADDED Requirements

### Requirement: Answer format

The generate prompt SHALL ask for the answer sentence first, followed by
its `[cN]` citations, and SHALL include one short worked example of that
format.

#### Scenario: Prompt carries the format and an example

- **GIVEN** retrieved chunks above the threshold
- **WHEN** the model is called to answer
- **THEN** the prompt it receives asks for the answer sentence first, then its `[cN]` citations, and contains one worked example of that format

### Requirement: Corrective retry before abstaining

When a reply is citation-only, or has answer text but no valid citation,
the system SHALL regenerate once with a corrective instruction naming the
failure; if the second reply still fails, it SHALL abstain. A reply
containing `NOT_IN_SOURCES` SHALL abstain without a retry. The system
SHALL never show a citation-only answer.

#### Scenario: Citation-only reply is retried

- **GIVEN** a retrieved chunk that contains the answer
- **WHEN** the model replies with citation markers only (e.g. `"[c1]"` or `"[40] [c1]"`)
- **THEN** the model is called once more with a corrective instruction, and a valid second reply is returned as the answer with its citations

#### Scenario: Citation-only twice abstains

- **GIVEN** the model replies with citation markers only, twice
- **WHEN** the question is answered
- **THEN** the system abstains with no citations, after exactly two model calls

#### Scenario: Answer without a valid citation is retried

- **GIVEN** a reply with answer text but no `[cN]` within the retrieved range
- **WHEN** the question is answered
- **THEN** the model is called once more with a corrective instruction; a second reply with a valid citation is returned, and otherwise the system abstains

#### Scenario: NOT_IN_SOURCES is not retried

- **GIVEN** the model replies `NOT_IN_SOURCES`
- **WHEN** the question is answered
- **THEN** the system abstains after exactly one model call

### Requirement: Deterministic generation

The chat model used for answers SHALL run with temperature 0.

#### Scenario: Chat model sampling

- **WHEN** the chat model is built from `PLUMBLINE_LLM`
- **THEN** its temperature is 0

### Requirement: Baseline models meet the trust gate

With the 7.4 baseline models on the reference M1, generation SHALL reach
correctness of at least 0.80 on `docx` and `pdf`, while faithfulness and
abstention stay at or above 0.80, with the abstain threshold, golden sets
and judge unchanged.

#### Scenario: Gate met

- **GIVEN** `ollama:llama3.1:8b` (chat), `ollama:nomic-embed-text` (embeddings), `ollama:qwen2.5:7b` (judge), threshold 0.3
- **WHEN** `plumbline-eval --connectors pdf,docx` runs, twice
- **THEN** both runs report correctness >= 0.80, faithfulness >= 0.80 and abstention >= 0.80 on each connector, compared with the 7.4 baseline in the eval plan
