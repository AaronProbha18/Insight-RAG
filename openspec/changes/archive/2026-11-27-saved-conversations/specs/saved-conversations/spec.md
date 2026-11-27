## ADDED Requirements

### Requirement: Conversations are stored

The system SHALL store each answered question in a conversation. A chat request without a conversation id SHALL create a conversation titled from its question; a request with one SHALL add to it. The stream's `done` event SHALL carry the conversation id.

#### Scenario: First question creates a conversation

- **GIVEN** no conversation id in the chat request
- **WHEN** the question is answered or abstained on
- **THEN** a conversation titled from the question exists, and the `done` event carries its id with the `answer_id`

#### Scenario: Follow-up joins the conversation

- **GIVEN** an existing conversation id in the chat request
- **WHEN** the question is answered
- **THEN** the answer is added to that conversation and the `done` event carries the same id

#### Scenario: Unknown conversation

- **GIVEN** a conversation id that does not exist
- **WHEN** a chat request names it
- **THEN** the response is 404 and no answer is generated or recorded

#### Scenario: Failed answer stores nothing

- **GIVEN** no conversation id
- **WHEN** answer generation fails
- **THEN** the stream is a single `error` event and no conversation is created

### Requirement: Conversations survive a reload

The system SHALL list conversations newest first and SHALL restore a conversation's questions, answers, abstentions and citations when it is opened at its own URL.

#### Scenario: Reload restores the chat

- **GIVEN** a chat with answers
- **WHEN** the person reloads the page
- **THEN** the conversation is still listed and opens with the same answers and citations

#### Scenario: Own URL

- **GIVEN** a saved conversation
- **WHEN** a browser loads `/app/chats/{id}` directly
- **THEN** that conversation opens

#### Scenario: Header shows title and scope

- **GIVEN** an open conversation and N indexed documents
- **WHEN** the chat page shows
- **THEN** the header shows the conversation title and the read-only label "Searching all N documents"

### Requirement: New chat keeps the previous conversation

New chat SHALL open an empty conversation and SHALL leave the previous one in the list.

#### Scenario: New chat

- **GIVEN** saved conversations
- **WHEN** the person presses New chat
- **THEN** an empty conversation opens and the previous one stays in the list

### Requirement: Rename and delete

The person SHALL be able to rename and delete a conversation from its sidebar row. Titles SHALL be 1-200 characters after trimming. Deleting a conversation SHALL NOT delete its recorded answers or their feedback.

#### Scenario: Rename

- **GIVEN** a conversation in the list
- **WHEN** the person renames it
- **THEN** the list and header show the new title

#### Scenario: Invalid title

- **GIVEN** a conversation
- **WHEN** a rename sends an empty or over-200-character title
- **THEN** the response is 422 and the title is unchanged

#### Scenario: Delete

- **GIVEN** a conversation in the list
- **WHEN** the person deletes it
- **THEN** the list updates, and its URL shows "Conversation not found"

#### Scenario: Feedback survives delete

- **GIVEN** a conversation with a rated answer
- **WHEN** the conversation is deleted
- **THEN** the answer and its feedback remain for the feedback export

### Requirement: Deleted sources are marked

When a conversation is opened, each citation SHALL say whether its document is still indexed, and the answer SHALL still show.

#### Scenario: Cited document deleted

- **GIVEN** a conversation whose cited document was deleted
- **WHEN** it is opened
- **THEN** the answer still shows, and the citation says the source is no longer indexed

### Requirement: Chats section in the sidebar

The sidebar SHALL show a **Chats** section listing conversations, and SHALL hide it when there are none. It SHALL be usable by keyboard and meet the shell's contrast rules.

#### Scenario: No conversations

- **GIVEN** no conversations
- **WHEN** the app opens
- **THEN** the Chats section is hidden

#### Scenario: Accessible list

- **GIVEN** the list
- **WHEN** used by keyboard, in light and dark mode
- **THEN** each row is reachable with an accessible name, the current one is announced (`aria-current="page"`), and text meets 4.5:1 contrast
