# answer-feedback Specification

## Purpose

Recording every answer, letting people rate it thumbs up or down with a reason, and exporting the ratings as golden-question candidates for review. Introduced by the `answer-feedback` change (issue #13).

## Requirements

### Requirement: Answers are identifiable

Every completed answer, abstentions included, SHALL be recorded with its
question, answer text, cited documents and locators, chat and embedding
model ids and abstain threshold, and the `/chat` `done` event SHALL carry
its public `answer_id`.

#### Scenario: done carries an answer id

- **GIVEN** a question that is answered or abstained
- **WHEN** the `/chat` stream ends
- **THEN** the `done` event carries an `answer_id` starting `ans_`, and the answer is recorded under it

### Requirement: Feedback on answers

`POST /feedback` SHALL store thumbs up or down for an answer, with an
optional reason and comment on thumbs down, keeping one record per answer.

#### Scenario: Thumbs up

- **GIVEN** a completed answer
- **WHEN** the person gives thumbs up
- **THEN** `POST /feedback` stores it, and the control shows that it was recorded

#### Scenario: Thumbs down with reason and comment

- **GIVEN** a completed answer
- **WHEN** the person gives thumbs down with the reason "wrong source" and a comment
- **THEN** the reason and comment are stored with the question, the answer text and the cited document ids and locators

#### Scenario: Feedback on an abstention

- **GIVEN** an abstention
- **WHEN** the person gives thumbs down with the reason "should have answered"
- **THEN** it is stored as feedback on an abstention

#### Scenario: Changing feedback

- **GIVEN** feedback already given on an answer
- **WHEN** the person changes it
- **THEN** the latest value replaces the earlier one, leaving one record per answer

### Requirement: Feedback errors reveal nothing

Invalid feedback SHALL return 422 and an unknown `answer_id` SHALL return
404, with no internal details. No response SHALL include retrieval scores
or confidence.

#### Scenario: Invalid or unknown

- **GIVEN** an invalid feedback request or an unknown `answer_id`
- **WHEN** it is sent
- **THEN** the API returns 422 or 404 with no internal details, and no response ever includes retrieval scores or confidence

### Requirement: Export as golden candidates

`plumbline-feedback-export` SHALL write every feedback record as a
candidate in the `golden.yaml` shape with `reference_answer` empty, and
SHALL NOT modify any golden set.

#### Scenario: Export

- **GIVEN** stored feedback
- **WHEN** a maintainer runs the export command
- **THEN** it writes candidates in the `golden.yaml` shape (question, answerable, supporting) with `reference_answer` left empty, and nothing is added to the golden sets automatically
