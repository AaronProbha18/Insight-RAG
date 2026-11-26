## MODIFIED Requirements

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
