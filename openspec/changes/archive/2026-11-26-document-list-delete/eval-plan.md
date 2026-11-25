## Applicability

Applies, narrowly. Retrieval, prompts and the graph are unchanged, so the
golden-set metrics can't move for an unchanged corpus. What can go wrong
is a deleted document still being retrieved or cited; that is checked by
a deterministic integration test, not a judge run. Product owner decided
not to re-run the full `plumbline-eval`.

## Dataset

No golden-set changes. The test uses synthetic text: two documents, one
of which alone contains the answer to a test question.

## Metrics

| Metric | Baseline | Target / Threshold |
| --- | --- | --- |
| Chunks retrieved from a deleted document | n/a (delete is new) | 0 |
| Citations to a deleted document | n/a | 0 |
| Abstains on a question only the deleted document answered | n/a | Yes |

## Pass Rule

Any retrieved chunk or citation from a deleted document, or no abstention
on the test question, blocks merge.

## How to Run

```bash
cd backend && uv run pytest tests/test_documents_api.py -k "deleted"
```

Uses real Postgres + pgvector retrieval with the test suite's fake
embeddings and chat model; the abstention decision comes from the real
graph and threshold.
