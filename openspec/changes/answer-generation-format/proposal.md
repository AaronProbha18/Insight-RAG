Issue: #26

## Why

The real-model baseline (rag-foundation 7.4, M1) fails the 0.80
correctness gate: docx 0.767, pdf 0.700. Retrieval isn't the cause
(hit-rate 0.967 / 1.0). Generation drops answers it had the source for:
- citation-only replies (`"[c1]"`) are shown as answers;
- replies with no valid citation become abstentions (6 on pdf);
- output varies from run to run.

## What Changes

- **Prompt:** asks for the answer sentence first, then its `[cN]`
  citations, with one short synthetic worked example.
- **Corrective retry (once):**
  - when a reply is **citation-only**, or **has answer text but no valid
    citation**, the graph regenerates once with a corrective instruction;
  - if the second reply still fails, the system abstains;
  - `NOT_IN_SOURCES` is never retried, because it's a legitimate refusal;
  - a citation-only answer is never shown.
- **Deterministic sampling:** the chat model runs at temperature 0.
- **Eval report:** for each answerable question, the report records the
  rank and similarity score of the best supporting chunk among those
  retrieved, or "not retrieved".
- **Eval runs:** the final eval on the reference M1 with the baseline
  models, plus a second identical run to measure run-to-run variation.
  Both are compared with the 7.4 baseline.

Scope gained in triage (owner decision): the retry also covers "answer
text but no valid citation", aimed at pdf's 6 abstentions with the
source retrieved. The second eval run is also new.

## Non-Goals

- Retrieval changes (chunk size, reranker, hybrid search).
- The abstain threshold (stays 0.3), golden sets, the judge model or its
  sampling.
- A larger chat model as the fix. One may be run only as a comparison.
- Live token streaming (D8 option A stays).

## Capabilities

### New

- none

### Modified

- `cited-answering`: answer format, a corrective retry, and deterministic
  generation.
- `answer-evaluation`: the report records the rank and score of the
  supporting chunk.

## Risk

| Area | Yes/No | Note |
| --- | --- | --- |
| Answer quality (retrieval, ranking, chunking, embeddings, prompts, graph) | Yes | Prompt, graph and sampling change; a real eval against the 7.4 baseline is required. |
| Connector / ingestion | No | |
| Auth / document ACLs | No | |
| Public API or stream contract | No | `/chat` events are unchanged; the eval report gains two per-question fields (additive). |
| Persistence / migration | No | No schema change; `answers` rows are unchanged. |
| New external dependency or service | No | |

## Impact

`backend/src/plumbline/graph.py`, `models.py`, `evals/scoring.py`
(and the runner output), backend tests, `docs/architecture/overview.md`
(the generate step), `evals/baselines/answer-generation-format/`. A retry
adds one LLM call, only when a reply fails the format check.
