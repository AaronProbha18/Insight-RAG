# answer-evaluation Specification

## Purpose

Golden sets, scoring, and the per-connector 80% gate. Introduced by the `rag-foundation` change.

## Requirements

### Requirement: Golden set per connector

Every supported connector SHALL have a fixture corpus and a golden set of
questions with reference answers and supporting passages, of which at
least 20% are unanswerable from the corpus. All content SHALL be
synthetic or permissively licensed public material with its source
recorded.

#### Scenario: Golden set shape is validated

- **WHEN** the eval runner loads a connector's golden set
- **THEN** it fails fast if any question lacks a reference answer or the unanswerable share is below 20%

### Requirement: Per-connector trust gate

The eval runner SHALL, for each connector, ingest its fixture corpus into
an isolated store, ask every golden question through the production
answer graph, and score answer correctness, faithfulness, abstention on
unanswerable questions, and retrieval hit-rate@k. It SHALL exit non-zero
if correctness, faithfulness, or abstention is below 0.80 for any
connector.

#### Scenario: Connector meets the bar

- **GIVEN** a connector scores correctness 0.86, faithfulness 0.91, abstention 0.83
- **WHEN** the eval runner completes
- **THEN** the connector is reported PASS and the exit code is 0

#### Scenario: Connector misses the bar

- **GIVEN** a connector scores faithfulness 0.74
- **WHEN** the eval runner completes
- **THEN** that connector is reported FAIL naming the metric, and the exit code is non-zero

### Requirement: Judge validation

The eval runner SHALL check the LLM judge against a set of human-labelled
cases before scoring, and SHALL refuse to report a gate result if judge
agreement with the human labels is below 0.80.

#### Scenario: Unreliable judge

- **GIVEN** the configured judge agrees with 6 of 10 human labels
- **WHEN** the eval runner starts
- **THEN** it stops with an error naming the judge model and its agreement rate, and reports no PASS/FAIL

### Requirement: Reproducible report

The eval runner SHALL write a report recording the LLM, embedding and
judge model identifiers, the golden set version, each metric per
connector, and the per-question results.

#### Scenario: Report written

- **WHEN** an eval run finishes
- **THEN** a report file exists containing the model identifiers and every metric per connector
