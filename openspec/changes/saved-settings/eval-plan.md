## Applicability

Applies. Top-k, threshold and chat model move from env/code into the
database. The default preset must reproduce today's behaviour exactly; a
wiring or seeding mistake would change answers.

## Dataset

Existing golden sets under `evals/connectors/{pdf,docx}`. No new cases.

## Metrics

| Metric | Baseline | Target / Threshold |
| --- | --- | --- |
| Retrieval metrics `plumbline-eval` reports | Post-rename run on the M1 (#11 task 5.1), same models and env | Exactly equal |
| Judge metrics (correctness, faithfulness, abstention) | Same run | Within the run-to-run spread recorded for #11; all >= 0.80 gates |
| Seeded default preset | n/a | top-k 5, threshold = env/default, chat model = `PLUMBLINE_LLM` |

## Pass Rule

Any retrieval metric differs from baseline, any judge metric outside the
spread, or any gate failing blocks merge. The run must also store its
score on `default@1` (checked with `plumbline-preset show default`).

## How to Run

```bash
# on the M1, same env as the #11 post-rename run
cd backend && uv run plumbline-eval --preset default --report ../evals/reports/presets.json
uv run plumbline-preset show default
```
