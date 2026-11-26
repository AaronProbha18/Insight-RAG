## MODIFIED Requirements

### Requirement: Document list

The Documents page SHALL list indexed documents with title, type and date
added, newest first, SHALL let the person search by title and filter by
type (All, PDF, Word), and SHALL show an empty state that points to upload
when there are none. Uploading or deleting a document SHALL update the list
and the sidebar's document count.

#### Scenario: Two documents

- **GIVEN** two indexed documents
- **WHEN** the person opens the Documents page
- **THEN** both appear with title, type and date added, newest first

#### Scenario: No documents

- **GIVEN** no indexed documents
- **WHEN** the Documents page opens
- **THEN** an empty state points to upload

#### Scenario: Search and filter

- **GIVEN** a PDF titled "Alpha" and a Word document titled "Beta"
- **WHEN** the person searches "alp", or filters by Word
- **THEN** only "Alpha", or only "Beta", is listed

#### Scenario: Count follows upload and delete

- **GIVEN** the Documents page
- **WHEN** the person uploads a file or deletes a document
- **THEN** each works as the dialog did before, and the sidebar's document count updates
