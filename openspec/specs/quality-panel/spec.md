# quality-panel Specification

## Purpose

What the chat UI shows of the latest eval result: per-connector PASS/FAIL against the 0.80 gates, the three gated metrics, run date, model ids and threshold; the not-measured and configuration-mismatch states; and the read-only summary endpoint behind it. Introduced by the `quality-panel` change (issue #14).

## Requirements

### Requirement: Quality panel shows the latest eval result

Each card on the Connectors page SHALL show that connector's correctness,
faithfulness and abstention, each as a number and a bar with the 0.80 mark,
labelled "Passes" or "Below bar" in a status pill against the 0.80 gates, and
a footer with the run date, the number of answerable and unanswerable
questions when the summary has them, the chat, embedding and judge model ids
and the abstain threshold.

#### Scenario: Both connectors passed

- **GIVEN** a summary where pdf and docx both passed
- **WHEN** the Connectors page opens
- **THEN** each card shows "Passes" with its three gated metrics, the run date and the model ids

#### Scenario: A gated metric failed

- **GIVEN** a quality summary where PDF is below the bar on correctness
- **WHEN** the Connectors page opens
- **THEN** the PDF card shows "Below bar" and names correctness as the metric under 0.80

#### Scenario: Question counts

- **GIVEN** a summary that has answerable and unanswerable question counts for pdf
- **WHEN** the Connectors page opens
- **THEN** the PDF card's footer says it was checked on that many answerable and unanswerable questions

#### Scenario: Summary without counts

- **GIVEN** a summary written before the counts existed
- **WHEN** the Connectors page opens
- **THEN** the cards show everything else and omit the question counts, with no error and no invented numbers

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

### Requirement: Read-only summary endpoint

`GET /quality` SHALL return summary fields only, including each connector's
answerable and unanswerable question counts when the summary has them as
non-negative integers, and `/chat` responses SHALL be unchanged.

#### Scenario: Summary fields only

- **GIVEN** the read-only endpoint for the summary
- **WHEN** called
- **THEN** it returns summary fields only (no per-question answers, file paths or internal ids), and `/chat` responses are unchanged

#### Scenario: Counts passed through

- **GIVEN** a summary whose pdf entry has `answerable_questions` and `unanswerable_questions`
- **WHEN** `GET /quality` is called
- **THEN** the pdf entry carries both numbers

#### Scenario: Counts absent or invalid

- **GIVEN** a summary without the counts, or with a count that is not a non-negative integer
- **WHEN** `GET /quality` is called
- **THEN** the entry omits the counts and the result is otherwise unchanged

### Requirement: Accessible panel

The Connectors page SHALL meet 4.5:1 text contrast in both themes, SHALL NOT
show Passes and Below bar by colour alone, and every link SHALL be reachable
by keyboard.

#### Scenario: Themes and keyboard

- **GIVEN** the Connectors page
- **WHEN** viewed in light and dark mode, by keyboard
- **THEN** Passes and Below bar are not shown by colour alone, text meets 4.5:1 contrast, and every link is reachable

### Requirement: Runner records question counts

The eval runner's summary SHALL record, for each connector, how many
answerable and how many unanswerable questions were scored.

#### Scenario: Counts in the summary

- **GIVEN** a run over a golden set of answerable and unanswerable questions
- **WHEN** the summary is written
- **THEN** each connector entry has `answerable_questions` and `unanswerable_questions` that add up to its question total
