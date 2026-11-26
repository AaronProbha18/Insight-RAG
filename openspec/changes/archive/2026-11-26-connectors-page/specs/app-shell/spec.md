## MODIFIED Requirements

### Requirement: Sidebar navigation

The app SHALL show a persistent left sidebar holding, top to bottom: the
Plumbline logo and a collapse button; New chat; a Knowledge group with
Documents (showing the document count) and "Connectors and quality", which
links to the Connectors page; and a link to the docs. There SHALL be no
separate Quality item, no Insights group, no quality state in the sidebar
and no quality dialog. The sidebar SHALL mark the current page.

#### Scenario: Open Documents from the sidebar

- **GIVEN** the app is open
- **WHEN** the person clicks Documents in the sidebar
- **THEN** the Documents page shows at `/app/documents`, and the sidebar marks Documents as the current page

#### Scenario: Open Connectors and quality from the sidebar

- **GIVEN** the app is open
- **WHEN** the person activates "Connectors and quality" in the sidebar
- **THEN** the Connectors page shows at `/app/connectors`, the sidebar marks that item as the current page, and no quality dialog exists

#### Scenario: Quality opens its dialog

- **GIVEN** this change has landed
- **WHEN** the person looks for the quality results in the sidebar
- **THEN** there is no Quality item, no Insights group and no quality dialog, and exactly one link, "Connectors and quality", goes to `/app/connectors`

### Requirement: Page URLs

Each page SHALL have its own URL (chat at `/`, Documents at
`/app/documents`, Connectors at `/app/connectors`). Browser back and forward
SHALL move between pages, and loading a page URL directly SHALL show that
page.

#### Scenario: Back returns to the chat

- **GIVEN** the person moved from the chat to Connectors
- **WHEN** they press the browser's back button
- **THEN** the chat shows at `/`

#### Scenario: Shared link in production

- **GIVEN** the backend serves the built UI
- **WHEN** a browser requests `/app/connectors` directly
- **THEN** the response is the UI's `index.html`, and the Connectors page shows

#### Scenario: API paths are not affected

- **GIVEN** the backend serves the built UI
- **WHEN** a client requests `GET /connectors`, `GET /documents` or an unknown path outside `/app/`
- **THEN** the API answers as before, and no `index.html` is returned for it
