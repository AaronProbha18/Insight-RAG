## Applicability

Applies. This change creates the answer pipeline and the first baseline
for every metric.

## Dataset

- `evals/connectors/pdf/` and `evals/connectors/docx/`, each with:
  - `corpus/` - 3-5 synthetic documents (e.g. a fictional company's
    policies, a product manual, meeting notes) written for this project,
    so licensing and privacy are clean. Content is the same facts across
    both formats where useful, so parser quality differences show up.
  - `golden.yaml` - 30-50 questions: `question`, `reference_answer`,
    `supporting` (document + locator), `answerable` (bool). At least 20%
    unanswerable; include multi-chunk and near-miss questions.
- `evals/judge/labelled.yaml` - ~10 human-labelled cases per judged metric
  (correctness, faithfulness) for judge validation.

## Metrics

| Metric | Baseline | Target / Threshold |
| --- | --- | --- |
| Answer correctness | none yet - this run establishes it | >= 0.80 per connector (gate) |
| Faithfulness | none yet | >= 0.80 per connector (gate) |
| Abstention on unanswerable | none yet | >= 0.80 per connector (gate) |
| Retrieval hit-rate@k (k = graph top_k) | none yet | tracked; recorded as baseline |
| Judge agreement with human labels | none yet | >= 0.80 or no gate result |
| Latency p95 per answer | none yet | tracked only |

## Pass Rule

The change passes when the judge validates (>= 0.80 agreement) and both
`pdf` and `docx` meet all three gated metrics. The abstain threshold (D7)
is tuned on the golden set once; record its value and the resulting
metrics in the report. Tuning further to hit the bar is allowed only by
changing pipeline parameters, never by editing golden answers.

## How to Run

```bash
docker compose up -d db
cd backend && uv run insight-rag-eval --connectors pdf,docx --report ../evals/reports/latest.json
```
