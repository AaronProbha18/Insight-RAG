## Applicability

N/A - display and API additions. The eval runner's summary file gains per-connector `answerable_questions` and `unanswerable_questions`; question scoring, the gate, retrieval, prompts and `/chat` are untouched, so no answer can change and no eval run is required. The runner's own tests cover the new fields.
