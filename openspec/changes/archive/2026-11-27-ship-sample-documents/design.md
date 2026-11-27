## Context

The sample corpus is four fictional Lumen Ridge documents, written as Markdown under `evals/corpus-src/` and rendered to PDF (`evals/connectors/pdf/corpus/`, about 4 KB each) and DOCX (`evals/connectors/docx/corpus/`, about 37 KB each). Both corpus folders also hold `corrupt/` fixtures. `.dockerignore` excludes all of `evals/`. `ingest()` already skips bytes whose sha256 `content_hash` is already stored and returns `created=False`, so re-ingesting the same file adds nothing.

## Decisions

### Ship the four PDFs only

- **Choice**: `evals/connectors/pdf/corpus/*.pdf` (top level only, which leaves out `corrupt/`).
- **Alternatives**: the DOCX copies (about 9× larger); both formats (same text with different hashes, so every passage would be indexed twice).
- **Why**: smallest files, the most familiar format, and page locators in citations. Four files matches the wireframe's "four short documents".
- **ADR**: none (local to this change)

### Copy at build time; keep the files where they are

- **Choice**: a `.dockerignore` re-include (`!evals/connectors/pdf/corpus/*.pdf`) plus `COPY evals/connectors/pdf/corpus/*.pdf /app/samples/`.
- **Alternatives**: move the samples to a shared `samples/` folder and repoint the connector evals and tests.
- **Why**: no churn in evals; the image gets the four files and nothing else.
- **UNVERIFIED**: that BuildKit honours a `!` re-include under an excluded `evals` entry. Confirm with `docker build` followed by `docker run --rm <img> ls -R /app/samples` and a check that `/app` has no `evals`. If it doesn't work, change `.dockerignore` from `evals` to `evals/*` plus specific re-includes.
- **ADR**: none

### Endpoint loops over `ingest()`

- **Choice**: `POST /documents/samples` reads every regular file directly in `PLUMBLINE_SAMPLES_DIR` (sorted by name) and calls `ingest()` on each. A per-file `ParseError` or `UnsupportedFileType` is recorded and the loop continues. Any other exception (database, embedding provider) propagates as a 5xx, the same as upload.
- **Alternatives**: a startup auto-index (rejected: the user should choose to load samples); a background job (rejected: overkill for four small files).
- **Why**: one ingestion path (issue OPEN QUESTION: reuse, yes), and idempotency comes from `content_hash`.
- **ADR**: none

### Samples directory resolution

- **Choice**: env `PLUMBLINE_SAMPLES_DIR`, set to `/app/samples` in the Dockerfile. When unset, it defaults to the repo's `evals/connectors/pdf/corpus`, resolved from `__file__` the same way `/quality` resolves its default. If the directory is missing, the endpoint returns 404 and names the path.
- **ADR**: none

## Interfaces

```http
POST /documents/samples            (no body)
200 OK
{
  "indexed": [ {"filename": "returns-policy.pdf", "document_id": "doc_…", "title": "…",
                "connector": "pdf", "chunks": 3, "created": true}, … ],
  "failed":  [ {"filename": "broken.pdf", "reason": "<ParseError message>"} ]
}
404 {"detail": "samples directory not found: <path>"}
```

The `indexed` items reuse the fields of `IngestResult` and add `filename`. The response is 200 even when some files are in `failed`; the caller checks that list.

## Data

none. This change uses the existing `documents.content_hash` uniqueness.

## Risks / Open Questions

- A partial failure returns 200, so the caller must read `failed`. #35 has to show it.
