## Applicability

Applies: the generate prompt, the graph's retry logic and the chat model's sampling change.

## Dataset

The existing golden sets, unchanged: `evals/connectors/docx/golden.yaml` and
`evals/connectors/pdf/golden.yaml` (40 questions each; 30 answerable). No
new cases; golden answers are never edited.

## Metrics

Baseline: rag-foundation 7.4, run A (threshold 0.3), M1, commit `476445a`,
evidence in `evals/baselines/rag-foundation/m1-7.4-t030.summary.json`.

| Metric | Baseline docx / pdf | Target / Threshold |
| --- | --- | --- |
| Answer correctness | 0.767 / 0.700 | >= 0.80 per connector, in **both** runs (gate) |
| Faithfulness | 0.897 / 0.875 | >= 0.80 per connector, in both runs (gate) |
| Abstention on unanswerable | 1.000 / 1.000 | >= 0.80 per connector, in both runs (gate) |
| Retrieval hit-rate@5 | 0.967 / 1.000 | tracked; should not move (retrieval unchanged) |
| Judge agreement with human labels | correctness 1.0, faithfulness 0.9 | >= 0.80, or no gate result |
| Latency p95 per answer | 61.0 s / 58.0 s | tracked only (the retry adds calls on failures) |
| Run-to-run spread (run 1 vs run 2, per metric) | not measured | recorded; no threshold |

Also recorded per run, as in 7.4: the failure-cause table (abstained with
the source retrieved, citation-only, wrong, retrieval missed), now with
`support_rank`/`support_score` for each failure.

## Pass Rule

Merge is blocked unless both runs:
- validate the judge (>= 0.80 agreement), and
- meet correctness, faithfulness and abstention >= 0.80 on both `docx`
  and `pdf`.

The models are the baseline's, with threshold 0.3 and top_k 5. Changing
the threshold, golden sets or judge to pass is not allowed. A larger chat
model may be run as a labelled comparison only; it never counts as the
pass.

## How to Run

On the reference M1 (Ollama native), with
`PLUMBLINE_LLM=ollama:llama3.1:8b`,
`PLUMBLINE_EMBEDDINGS=ollama:nomic-embed-text`,
`PLUMBLINE_JUDGE_LLM=ollama:qwen2.5:7b` and
`PLUMBLINE_EMBEDDING_DIM=768`:

```bash
docker compose up -d db
cd backend
uv run plumbline-eval --connectors pdf,docx --report ../evals/reports/agf-run1.json
uv run plumbline-eval --connectors pdf,docx --report ../evals/reports/agf-run2.json
```

Commit both `.summary.json` files to `evals/baselines/answer-generation-format/`.
