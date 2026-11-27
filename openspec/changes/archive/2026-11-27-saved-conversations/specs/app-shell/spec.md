## MODIFIED Requirements

### Requirement: Chat survives navigation

The current conversation SHALL stay open while the person moves between pages.
New chat SHALL open an empty conversation and SHALL keep the previous one in
the Chats list.

#### Scenario: Return to an unchanged chat

- **GIVEN** a chat with answers
- **WHEN** the person opens Documents and then returns to the chat
- **THEN** the chat is unchanged

#### Scenario: New chat clears it

- **GIVEN** a chat with answers
- **WHEN** the person activates New chat
- **THEN** the chat page shows an empty thread, and the previous conversation stays in the Chats list
