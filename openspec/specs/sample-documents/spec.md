# sample-documents Specification

## Purpose
TBD - created by archiving change ship-sample-documents. Update Purpose after archive.
## Requirements
### Requirement: Sample files ship in the image

The Docker image SHALL contain the sample documents at `/app/samples` and SHALL NOT contain golden sets, baselines or any other part of `evals/`.

#### Scenario: Samples present at the documented path

- **GIVEN** the built Docker image
- **WHEN** the container starts
- **THEN** the four sample PDFs are present at `/app/samples`

#### Scenario: No eval data in the image

- **GIVEN** the image
- **WHEN** it is inspected
- **THEN** it contains the sample files and no golden sets, baselines or `corrupt/` fixtures

### Requirement: Index samples on request

The system SHALL provide `POST /documents/samples`, which indexes every file in the samples directory through the normal ingestion path and returns the indexed and failed files.

#### Scenario: All samples indexed

- **GIVEN** no indexed documents
- **WHEN** the sample-index endpoint is called
- **THEN** every sample is indexed and listed by `GET /documents`

#### Scenario: Idempotent

- **GIVEN** samples already indexed
- **WHEN** the endpoint is called again
- **THEN** no document or passage is duplicated, and each sample is reported with `created: false`

#### Scenario: One sample fails to parse

- **GIVEN** a samples directory with one file that fails to parse
- **WHEN** the endpoint runs
- **THEN** the response's `failed` list names that file and the reason, and the other samples are still indexed

#### Scenario: Samples directory missing

- **GIVEN** `PLUMBLINE_SAMPLES_DIR` points to a missing directory
- **WHEN** the endpoint is called
- **THEN** it returns 404 naming the path

