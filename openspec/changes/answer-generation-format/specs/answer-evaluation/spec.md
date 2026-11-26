## MODIFIED Requirements

### Requirement: Reproducible report

The eval runner SHALL write a report recording the LLM, embedding and
judge model identifiers, the golden set version, each metric per
connector, and the per-question results. For each answerable question,
the per-question results SHALL include the rank and similarity score of
the best supporting chunk among those retrieved, or record that it was
not retrieved.

#### Scenario: Report written

- **WHEN** an eval run finishes
- **THEN** a report file exists containing the model identifiers and every metric per connector

#### Scenario: Supporting chunk located

- **GIVEN** an answerable question whose supporting chunk was retrieved at position 2 with score 0.61
- **WHEN** it is scored
- **THEN** its result records `support_rank` 2 and `support_score` 0.61

#### Scenario: Supporting chunk not retrieved

- **GIVEN** an answerable question whose supporting chunk was not among those retrieved
- **WHEN** it is scored
- **THEN** its result records `support_rank` and `support_score` as null
