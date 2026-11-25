# quality-panel Specification

## Purpose

What the chat UI shows of the latest eval result: per-connector PASS/FAIL against the 0.80 gates, the three gated metrics, run date, model ids and threshold; the not-measured and configuration-mismatch states; and the read-only summary endpoint behind it. Introduced by the `quality-panel` change (issue #14).

## Requirements

### Requirement: Quality panel shows the latest eval result

The chat UI SHALL show, per connector, PASS or FAIL against the 0.80
gates, the correctness, faithfulness and abstention values, and once the
run date, the chat, embedding and judge model ids and the abstain
threshold.

#### Scenario: Both connectors passed

- **GIVEN** a report where pdf and docx both passed
- **WHEN** the person opens the quality panel
- **THEN** each connector shows PASS with its three gated metrics, the run date and the model ids

#### Scenario: A gated metric failed

- **GIVEN** a report where docx failed faithfulness
- **WHEN** the panel opens
- **THEN** docx shows FAIL and names the metric below 0.80

### Requirement: Not measured state

When there is no summary, the run stopped at the judge gate (exit 2), or
the run was sampled, the panel SHALL say "Not measured yet", show no
numbers, and show the command to run an eval.

#### Scenario: No valid run

- **GIVEN** no report, or a run that stopped because the judge wasn't validated (exit 2)
- **WHEN** the panel opens
- **THEN** it says "Not measured yet", shows no numbers, and shows the command to run an eval

#### Scenario: Sampled run

- **GIVEN** a summary from a run with `--sample`
- **WHEN** the panel opens
- **THEN** it says "Not measured yet" and shows no numbers

### Requirement: Configuration mismatch warning

When the evaluated chat model, embedding model or abstain threshold
differs from the running configuration, the panel SHALL warn that the
result is for a different configuration.

#### Scenario: Different configuration

- **GIVEN** a report made with a different chat model, embedding model or abstain threshold than the app is running
- **WHEN** the panel opens
- **THEN** it warns that the result is for a different configuration

### Requirement: Read-only summary endpoint

`GET /quality` SHALL return summary fields only, and `/chat` responses
SHALL be unchanged.

#### Scenario: Summary fields only

- **GIVEN** the new read-only endpoint for the summary
- **WHEN** called
- **THEN** it returns summary fields only (no per-question answers, file paths or internal ids), and `/chat` responses are unchanged

### Requirement: Accessible panel

The panel SHALL meet 4.5:1 text contrast in both themes, SHALL NOT show
PASS and FAIL by colour alone, and SHALL be reachable by keyboard.

#### Scenario: Themes and keyboard

- **GIVEN** the panel
- **WHEN** viewed in light and dark mode, by keyboard
- **THEN** text meets 4.5:1 contrast, PASS and FAIL are not shown by colour alone, and the panel is reachable by keyboard
