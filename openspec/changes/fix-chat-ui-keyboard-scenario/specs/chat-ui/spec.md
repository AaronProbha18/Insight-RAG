## MODIFIED Requirements

### Requirement: Accessible, responsive, themed

The chat SHALL give every control an accessible name and a visible focus indicator, work by keyboard alone, have no horizontal scroll at 375px width, follow the OS light/dark preference, and meet WCAG AA (>= 4.5:1) for every text/background pair in both themes.

#### Scenario: Phone width in either theme

- **GIVEN** a 375px-wide viewport with the OS set to light or dark
- **WHEN** the user uses the chat
- **THEN** there is no horizontal scroll and every text/background pair meets 4.5:1

#### Scenario: Keyboard-only use

- **GIVEN** a keyboard-only user
- **WHEN** they ask a question, stop, copy, regenerate and open a source
- **THEN** every step is reachable with a visible focus indicator
