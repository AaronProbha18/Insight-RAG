## MODIFIED Requirements

### Requirement: Quality panel shows the latest eval result

Each card on the Connectors page SHALL show that connector's correctness,
faithfulness and abstention, each as a number and a bar with the 0.80 mark,
labelled "Passes" or "Below bar" against the 0.80 gates, and the run date,
the chat, embedding and judge model ids and the abstain threshold.

#### Scenario: Both connectors passed

- **GIVEN** a summary where pdf and docx both passed
- **WHEN** the Connectors page opens
- **THEN** each card shows "Passes" with its three gated metrics, the run date and the model ids

#### Scenario: A gated metric failed

- **GIVEN** a quality summary where PDF is below the bar on correctness
- **WHEN** the Connectors page opens
- **THEN** the PDF card shows "Below bar" and names correctness as the metric under 0.80

### Requirement: Not measured state

When there is no summary, the run stopped at the judge gate (exit 2), the
run was sampled, or the summary has no entry for a connector, that card
SHALL say "Not measured", show no numbers, and show the command to run an
eval.

#### Scenario: No valid run

- **GIVEN** no quality summary, or a run that stopped because the judge wasn't validated (exit 2)
- **WHEN** the Connectors page opens
- **THEN** each card says "Not measured" and shows how to run an eval

#### Scenario: Sampled run

- **GIVEN** a summary from a run with `--sample`
- **WHEN** the Connectors page opens
- **THEN** each card says "Not measured" and shows how to run an eval

### Requirement: Configuration mismatch warning

When the evaluated chat model, embedding model or abstain threshold differs
from the running configuration, the cards SHALL say the result is not
current.

#### Scenario: Different configuration

- **GIVEN** a summary made with a different model or threshold than the app runs
- **WHEN** the Connectors page opens
- **THEN** the cards say the result is not current

### Requirement: Accessible panel

The Connectors page SHALL meet 4.5:1 text contrast in both themes, SHALL NOT
show Passes and Below bar by colour alone, and every link SHALL be reachable
by keyboard.

#### Scenario: Themes and keyboard

- **GIVEN** the Connectors page
- **WHEN** viewed in light and dark mode, by keyboard
- **THEN** Passes and Below bar are not shown by colour alone, text meets 4.5:1 contrast, and every link is reachable
