## ADDED Requirements

### Requirement: Feedback controls on answers

Each completed assistant message, abstentions included, SHALL offer
thumbs up and thumbs down; thumbs down SHALL offer an optional reason and
comment; the control SHALL show the recorded value. The controls SHALL be
keyboard-usable with accessible names, visible focus rings, and text
meeting 4.5:1 contrast in both themes.

#### Scenario: Keyboard only

- **GIVEN** the feedback controls
- **WHEN** used by keyboard only
- **THEN** they have accessible names and visible focus rings, and text meets 4.5:1 contrast in both themes

#### Scenario: No controls while streaming or on errors

- **GIVEN** an assistant message that is still running or ended with an error
- **WHEN** it is shown
- **THEN** no feedback controls are shown for it
