## Applicability

Applies - regression check only. No retrieval logic changes; the eval
proves the rename, env-var plumbing and regenerated fixtures are no-ops.

## Dataset

Existing golden sets under `evals/connectors/{pdf,docx}`. No new cases.
Corpus regenerated from `evals/corpus-src` with the new author string.

## Metrics

| Metric | Baseline | Target / Threshold |
| --- | --- | --- |
| Every metric `insight-rag-eval` reports (hit-rate@k, MRR, faithfulness, citation correctness, abstention) | Captured twice on `origin/main` in task 1.1 (`evals/reports/pre-rename-{a,b}.json`) | Retrieval metrics: exactly equal. Judge metrics: within the spread between the two baseline runs (the runner has no tolerance of its own) |

## Pass Rule

Any retrieval metric differs from baseline, or any judge metric falls
outside the baseline run-to-run spread -> blocks merge. Connector conformance
failure -> blocks merge.

## How to Run

```bash
# baseline (before rename, on origin/main)
cd backend && uv run insight-rag-eval --report ../evals/reports/pre-rename-a.json
uv run insight-rag-eval --report ../evals/reports/pre-rename-b.json
# after rename
cd backend && uv run plumbline-eval --report ../evals/reports/post-rename.json
```
