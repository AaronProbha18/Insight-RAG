# document-management Specification

## Purpose

Listing and deleting indexed documents, through the API and the chat UI: what a person sees, what deletion removes, and that a deleted document is never used again. Introduced by the `document-list-delete` change (issue #12).

## Requirements

### Requirement: Document list

The chat UI SHALL list indexed documents with title, type and date added,
newest first, and SHALL show an empty state that points to upload when
there are none.

#### Scenario: Two documents

- **GIVEN** two indexed documents
- **WHEN** the person opens the document list
- **THEN** both appear with title, type and date added, newest first

#### Scenario: No documents

- **GIVEN** no indexed documents
- **WHEN** the list opens
- **THEN** an empty state points to upload

### Requirement: Delete a document

The system SHALL provide `DELETE /documents/{document_id}`. It SHALL
remove the document and all of its chunks in one transaction and return
204. The UI SHALL ask for confirmation before calling it.

#### Scenario: Delete an indexed document

- **GIVEN** an indexed document
- **WHEN** the person deletes it and confirms
- **THEN** `DELETE /documents/{document_id}` returns 204, and the document and all of its chunks are gone

#### Scenario: Unknown or already-deleted id

- **GIVEN** a `document_id` that doesn't exist or was already deleted
- **WHEN** delete is called
- **THEN** the API returns 404 with no internal details

### Requirement: Deleted documents are never used again

After deletion, retrieval SHALL NOT return the document's chunks and
answers SHALL NOT cite it.

#### Scenario: Question only the deleted document answered

- **GIVEN** a deleted document that alone answered a question
- **WHEN** that question is asked
- **THEN** the assistant abstains and cites nothing from the deleted document

### Requirement: Re-upload after delete

Uploading the same bytes after a delete SHALL index the file again.

#### Scenario: Same bytes again

- **GIVEN** a file that was deleted
- **WHEN** the same bytes are uploaded again
- **THEN** it is indexed again and the API returns 201

### Requirement: Accessible list and delete controls

Every list and delete control SHALL have an accessible name and a visible
focus ring, delete SHALL ask for confirmation, and text SHALL meet 4.5:1
contrast in both themes.

#### Scenario: Keyboard only

- **GIVEN** the list and its delete controls
- **WHEN** used by keyboard only
- **THEN** every control has an accessible name and a visible focus ring, delete asks for confirmation, and text meets 4.5:1 contrast in both themes
