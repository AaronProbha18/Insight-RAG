## Context

`graph.generate` ([graph.py](../../../backend/src/plumbline/graph.py))
makes one `llm.invoke(PROMPT)`, then:
- `NOT_IN_SOURCES` in the reply -> abstain;
- no `[cN]` with N in range -> abstain;
- otherwise the reply is the answer.

`"[c1]"` alone passes the last check, so a citation-only reply is shown.
`models.chat_model()` passes no sampling options, so the provider's
default temperature applies. The judge (`judge_model()`) is out of scope.

In the 7.4 baseline (run A, out of 30 answerable questions per connector):

| Cause | docx | pdf |
| --- | --- | --- |
| Abstained with the source retrieved | 0 | 6 |
| Citation-only reply | 4 | 1 |
| Wrong or unsupported | 2 | 2 |
| Abstained, retrieval missed | 1 | 0 |

Owner decisions in triage: the retry also covers "answer text, no valid
citation" (Q1a); the final eval runs twice to measure variation (Q2a).

## Decisions

### Answer-first prompt with a worked example

- **Choice**: `PROMPT` asks for a direct answer sentence first, followed
  by its `[cN]` citations. It adds one short synthetic example, e.g.
  sources `[c1] ... The office opens at 9am.` ->
  `The office opens at 9am [c1].` It keeps the `NOT_IN_SOURCES` rule
  unchanged.
- **Alternatives**: structured output (JSON with answer and citations).
  Small local models break JSON more often than prose, and the stream
  contract carries text.
- **Why**: the acceptance criteria; the example targets the citation-only
  replies.
- **ADR**: none (local to this change; guide rule 5.3 applies: eval
  required).

### Format check and one corrective retry

- **Choice**: after a reply, classify it:
  - contains `NOT_IN_SOURCES` -> **abstain** (no retry);
  - **citation-only**: after removing every bracketed token (`\[[^\]]*\]`,
    which covers `[c1]` and `[40]`) and whitespace/punctuation, no letter
    or digit remains -> **retry**;
  - **no valid citation**: answer text, but no `[cN]` with
    `1 <= N <= len(chunks)` -> **retry**;
  - otherwise -> **answer**.

  The retry is one more `llm.invoke`: the original prompt, plus the bad
  reply and a corrective instruction naming the failure ("Your reply had
  no answer text" / "Your reply cited no source"). The second reply is
  classified again, and anything other than **answer** -> abstain. There
  is never a third call.
- **Alternatives**:
  - abstain immediately (today's behaviour; it loses the answers);
  - strip citations and show the text (breaks the trust rule);
  - retry more than once (latency; the issue says once).
- **Why**: the trust rule is kept. An answer shown always has answer text
  and a valid citation.
- **ADR**: none

### Temperature 0 for the chat model

- **Choice**: `models.chat_model()` calls
  `init_chat_model(..., temperature=0)`. `judge_model()` is unchanged.
- **UNVERIFIED** - confirm that `init_chat_model` forwards `temperature`
  to `ChatOllama`, by building the model in a test and asserting
  `.temperature == 0`.
- **Why**: the acceptance criteria; it removes chat-side variation. Judge
  variation remains, which is what the second eval run measures.

### Supporting-chunk rank and score in the report

- **Choice**: `score_question` already finds whether a supporting chunk
  was retrieved (`hit`). For answerable items it also records
  `support_rank` (the 1-based position of the first retrieved chunk
  matching a golden `(document, locator)`) and `support_score` (that
  chunk's similarity), or `None`/`None` when it wasn't retrieved. Both
  appear per question in the full report. The `.summary.json` is
  unchanged.
- **Why**: splits failures into retrieval and generation causes.

## Interfaces

```python
# graph.py
def _classify(reply: str, n_chunks: int) -> Literal["abstain", "retry_no_text", "retry_no_citation", "answer"]: ...

# evals/scoring.py
@dataclass(frozen=True)
class QuestionScore:
    ...
    support_rank: int | None = None   # 1-based; None = not retrieved / unanswerable
    support_score: float | None = None
```

## Data

None.

## Risks / Open Questions

- The retry adds about one call's latency (roughly 30 to 60 s on the M1)
  to the questions that fail the check. Latency p95 is tracked, not gated.
- If the final eval still misses 0.80, verify fails. The threshold,
  golden sets and judge don't move; a follow-up issue decides next steps
  using the new rank and score data.
