## Applicability

Applies - regression check only. No retrieval logic changes; the eval
proves the rename, env-var plumbing and regenerated fixtures are no-ops.

## Dataset

Existing golden sets under `evals/connectors/{pdf,docx}`. No new cases.
Corpus regenerated from `evals/corpus-src` with the new author string.

## Metrics

| Metric | Baseline | Target / Threshold |
| --- | --- | --- |
| hit_rate, abstention (retrieval-based) | `evals/reports/pre-rename-a.json`, one run on unrenamed code | Exactly equal |
| correctness, faithfulness (judge-based) | same report | Within +/-0.05 (absolute) of the baseline, per connector |

Models for baseline and post-rename run are identical: chat
`ollama:llama3.1:8b`, embeddings `ollama:nomic-embed-text`, judge
`ollama:qwen2.5:7b`. The baseline ran on a Windows CPU machine, the
post-rename run on an Apple M1, so retrieval equality across machines is
UNVERIFIED; if it differs, record the difference and decide with the
product owner whether it counts as a regression.

The baseline itself is below the 0.80 correctness gate on both connectors
(docx 0.67, pdf 0.73). This change is judged against the baseline, not
against 0.80; `verification.md` states the gate result separately.

## Pass Rule

Any retrieval metric differs from baseline, or any judge metric moves more
than 0.05 from baseline -> blocks merge. Connector conformance failure ->
blocks merge.

## How to Run

```bash
# env for both runs (README defaults, judge per the Metrics section):
#   INSIGHT_RAG_/PLUMBLINE_ LLM=ollama:llama3.1:8b EMBEDDINGS=ollama:nomic-embed-text
#   EMBEDDING_DIM=768 JUDGE_LLM=ollama:qwen2.5:7b
# baseline (before rename, on the unrenamed code)
cd backend && uv run insight-rag-eval --report ../evals/reports/pre-rename-a.json
# after rename
cd backend && uv run plumbline-eval --report ../evals/reports/post-rename.json
```
