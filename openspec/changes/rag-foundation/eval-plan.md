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

Baseline = task 7.4 run A (threshold 0.3), docx / pdf. See
[Baseline run](#baseline-run-task-74) below.

| Metric | Baseline | Target / Threshold |
| --- | --- | --- |
| Answer correctness | **0.767 / 0.700** (FAIL) | >= 0.80 per connector (gate) |
| Faithfulness | 0.897 / 0.875 | >= 0.80 per connector (gate) |
| Abstention on unanswerable | 1.000 / 1.000 | >= 0.80 per connector (gate) |
| Retrieval hit-rate@k (k = graph top_k) | 0.967 / 1.000 (k = 5) | tracked; recorded as baseline |
| Judge agreement with human labels | correctness 1.0, faithfulness 0.9 | >= 0.80 or no gate result |
| Latency p95 per answer | 61.0 s / 58.0 s (M1, Ollama native) | tracked only |

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

## Baseline run (task 7.4)

Run on the reference M1 (arm64, Ollama native) on 2026-11-25 at commit
`476445a`, 40 golden questions per connector. Models:
`ollama:llama3.1:8b` (chat), `ollama:nomic-embed-text` (embeddings),
`ollama:qwen2.5:7b` (judge). Evidence:
[`evals/baselines/rag-foundation/`](../../../evals/baselines/rag-foundation/)
(shareable summaries and the smoke output).

| Run | Threshold | Correctness docx / pdf | Faithfulness | Abstention | Hit-rate | Duration |
| --- | --- | --- | --- | --- | --- | --- |
| A (baseline) | 0.3 | 0.767 / 0.700 | 0.897 / 0.875 | 1.0 / 1.0 | 0.967 / 1.0 | 61 min |
| B (tuned once) | 0.2 | 0.800 / 0.667 | 0.897 / 0.870 | 1.0 / 1.0 | 0.967 / 1.0 | 42 min |

**Threshold decision: keep 0.3.** The one allowed tuning did not move
correctness: the mean is 0.733 at both thresholds. Failures per run, out
of 30 answerable questions per connector:

| Cause | A docx | A pdf | B docx | B pdf |
| --- | --- | --- | --- | --- |
| Abstained although the right source was retrieved (`hit=True`) | 0 | 6 | 0 | 7 |
| Abstained, retrieval missed (`hit=False`) | 1 (q21) | 0 | 1 (q21) | 0 |
| Citation-only reply, no answer text (`"[c1]"`, `"[40] [c1]"`) | 4 | 1 | 3 | 0 |
| Answered with wrong or unsupported content | 2 | 2 | 2 | 3 |
| **Correct** | **23** | **21** | **24** | **20** |

So the threshold is not the lever. Retrieval works (hit-rate 0.97 to 1.0);
the losses are in answer generation. Either the model says
`NOT_IN_SOURCES` or omits a valid citation, so the graph discards an
answer it had the source for (PDF), or it emits only citation markers
(DOCX). Different questions fail in A and B, including failures a
threshold change cannot cause (abstentions with the source retrieved,
citation-only replies), which points to **run-to-run variation in the
model's output**. Its size is not measured yet: repeat a run at fixed
settings before treating differences of a few hundredths as real.

**Verdict: the pass rule is not met.** The judge validates, faithfulness,
abstention and retrieval pass, but correctness is below 0.80 on both
connectors in run A (and on pdf in run B). Per the pass rule this is not
fixed by further threshold tuning or by editing golden answers; the fix
belongs to answer generation (prompt and output format, deterministic
sampling) in a follow-up `answer-quality` change, measured against this
baseline.

Smoke test (task 9.2) on the same M1, via Docker: health, upload (201,
3 chunks), and a cited answer all PASS.
