## ADDED Requirements

### Requirement: Upload and index a document

The system SHALL accept a PDF or DOCX upload, parse it with the matching
connector, split it into chunks that keep their source locator, embed the
chunks, and store the document and its chunks atomically.

#### Scenario: PDF is indexed

- **WHEN** a valid PDF is uploaded
- **THEN** the response is 201 with a public `document_id`, title, connector `pdf`, chunk count > 0, and `created: true`
- **AND** every stored chunk has a locator naming its page

#### Scenario: DOCX is indexed

- **WHEN** a valid DOCX is uploaded
- **THEN** the response is 201 with connector `docx` and chunk count > 0

### Requirement: Idempotent ingestion

The system SHALL identify documents by the SHA-256 of their bytes and SHALL
NOT create a second copy of an identical file.

#### Scenario: Same file uploaded twice

- **GIVEN** a file was already ingested
- **WHEN** the identical bytes are uploaded again, under any filename
- **THEN** the response is 200 with the original `document_id` and `created: false`, and no new chunks are stored

### Requirement: No partial documents

The system SHALL NOT leave a document stored without its chunks if any
ingestion step fails.

#### Scenario: Embedding fails mid-ingestion

- **GIVEN** the embedding provider errors during ingestion
- **WHEN** a file is uploaded
- **THEN** the request fails with an error response and neither the document nor any of its chunks is stored

### Requirement: Reject unsupported or unreadable files

The system SHALL reject files no registered connector supports, and files
the connector cannot parse, without storing anything.

#### Scenario: Unsupported type

- **WHEN** a `.png` file is uploaded
- **THEN** the response is 415 naming the supported types

#### Scenario: Corrupt PDF

- **WHEN** a file with a `.pdf` extension but invalid content is uploaded
- **THEN** the response is 422 and nothing is stored

### Requirement: List documents

The system SHALL list indexed documents with public id, title, connector,
and creation time, and SHALL NOT expose internal database ids.

#### Scenario: Listing after uploads

- **GIVEN** two documents were ingested
- **WHEN** the document list is requested
- **THEN** both appear with `document_id` values of the form `doc_<ULID>`
