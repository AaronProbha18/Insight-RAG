## ADDED Requirements

### Requirement: Default preset

When no presets exist at startup, the system SHALL create a preset from
today's values (chat model, abstain threshold, top-k) as version 1 and
make it active.

#### Scenario: First start

- **GIVEN** no presets exist
- **WHEN** the app starts
- **THEN** a default preset is created from today's values and is active

### Requirement: Versioned presets

Saving a preset SHALL store its top-k, abstain threshold and chat model.
A new name SHALL be stored as version 1; changed settings for an existing
name SHALL be stored as a new version, leaving earlier versions readable
and unchanged.

#### Scenario: New preset

- **GIVEN** a preset name and settings
- **WHEN** it is saved
- **THEN** it is stored in the database as version 1

#### Scenario: Changed settings

- **GIVEN** an existing preset
- **WHEN** its settings are changed
- **THEN** a new version is stored and earlier versions remain readable and unchanged

### Requirement: Validated settings

The system SHALL reject a save with a reason when top-k is below 1 or
above the maximum, the threshold is outside 0..1, or the chat model is
not in the allowlist.

#### Scenario: Out-of-range settings

- **GIVEN** settings outside allowed ranges (e.g. top-k < 1, threshold outside 0..1, unknown model)
- **WHEN** saved
- **THEN** the save is rejected with a reason

### Requirement: Answers use the active preset

Retrieval and generation SHALL use the active preset's top-k, threshold
and chat model. Chunking and the embedding model are index settings and
SHALL NOT be part of a preset.

#### Scenario: Answering with the active preset

- **GIVEN** an active preset
- **WHEN** a question is answered
- **THEN** retrieval and generation use that preset's top-k, threshold and models

### Requirement: Last eval score per version

An eval run against a preset version SHALL store its score and run time
on that version, replacing the previous score. Sampled and judge-stopped
runs SHALL NOT store a score.

#### Scenario: Eval finishes

- **GIVEN** an eval run against a preset version
- **WHEN** it finishes
- **THEN** that version stores the score and run time, replacing its previous score

#### Scenario: Never evaluated

- **GIVEN** a preset version never evaluated
- **WHEN** listed
- **THEN** it shows "not evaluated", not a score

### Requirement: Listing presets

Listing SHALL show, for each preset, its name, version, settings and last
eval score.

#### Scenario: List

- **GIVEN** presets exist
- **WHEN** listed
- **THEN** each shows name, version, settings and last eval score

### Requirement: Documents record their chunking

Each document SHALL record the chunk size and overlap it was indexed
with.

#### Scenario: Upload

- **GIVEN** a file is uploaded
- **WHEN** it is indexed
- **THEN** its document record holds the chunk size and overlap used
