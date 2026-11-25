## Applicability

Applies - regression check only. No retrieval logic changes; the eval
proves the rename, env-var plumbing and regenerated fixtures are no-ops.

## Dataset

Existing golden sets under `evals/connectors/{pdf,docx}`. No new cases.
Corpus regenerated from `evals/corpus-src` with the new author string.

## Metrics

Baseline: `evals/baselines/rag-foundation/m1-7.4-t030.summary.json` (rag-foundation
task 7.4: full run on the reference Apple M1 at commit `476445a`, threshold 0.3,
top_k 5, no `--sample`). The post-rename run is on the same M1 with the same
models, so the comparison is like-for-like.

| Metric | Baseline | Target / Threshold |
| --- | --- | --- |
| hit_rate, abstention (retrieval-based) | docx 0.967 / 1.0, pdf 1.0 / 1.0 | Exactly equal |
| correctness, faithfulness (judge-based) | docx 0.767 / 0.897, pdf 0.700 / 0.875 | Within +/-0.05 (absolute) of the baseline, per connector |
| models, threshold, top_k, sample, `golden_version` per connector | in the summary | Equal |

Models for baseline and post-rename run are identical: chat
`ollama:llama3.1:8b`, embeddings `ollama:nomic-embed-text`, judge
`ollama:qwen2.5:7b`. That the post-rename run uses the same physical M1 as the
baseline is assumed; if it does not, say so in `verification.md`.

The baseline itself is below the 0.80 correctness gate on both connectors
(docx 0.77, pdf 0.70). This change is judged against the baseline, not
against 0.80; `verification.md` states the gate result separately.

Supplementary only: `evals/reports/pre-rename-a.json`, a CPU run on Windows
with the same models. Its judge metrics differ from the M1 baseline by up to
0.10 (docx correctness) on equivalent code, which is why it is not the
comparison baseline. Its `golden_version` hashes differ from the M1 ones only
because the Windows checkout has CRLF line endings; the golden files are
identical in git.

## Pass Rule

Any retrieval metric differs from baseline, or any judge metric moves more
than 0.05 from baseline -> blocks merge. Connector conformance failure ->
blocks merge.

## How to Run

```bash
# on the M1, from backend/, same models as the baseline
export PLUMBLINE_LLM=ollama:llama3.1:8b PLUMBLINE_EMBEDDINGS=ollama:nomic-embed-text        PLUMBLINE_EMBEDDING_DIM=768 PLUMBLINE_JUDGE_LLM=ollama:qwen2.5:7b
uv run plumbline-eval --report ../evals/reports/post-rename.json   # also writes post-rename.summary.json
```
