Issue: #40

## Why

A new user has nothing to try: the fictional Lumen Ridge sample corpus lives only under `evals/`, which the Docker image excludes. The first-run "Try the sample documents" action (#35) needs the samples in the image and a way to index them.

## What Changes

- The Docker image carries the four sample PDFs (`evals/connectors/pdf/corpus/*.pdf`) at `/app/samples`. Nothing else from `evals/` goes in, including `corrupt/`.
- New endpoint `POST /documents/samples` indexes every file in the samples directory through the normal `ingest()` path and reports per-file results and failures.
- `PLUMBLINE_SAMPLES_DIR` sets the samples directory. In local dev it defaults to the repo's PDF corpus.
- Docs record the path, the env var and the endpoint.

## Non-Goals

- New sample content.
- The first-run screen (#35).
- Bulk removal of samples; the normal document delete still applies.
- Tagging samples apart from the user's own documents (issue OPEN QUESTION, resolved: no).
- Shipping the DOCX versions of the corpus: they hold the same text, so indexing both formats would duplicate every passage.

## Capabilities

### New

- `sample-documents`: sample files shipped in the image, plus an idempotent endpoint that indexes them.

### Modified

- none

## Risk

| Area | Yes/No | Note |
| --- | --- | --- |
| Answer quality (retrieval, ranking, chunking, embeddings, prompts, graph) | No | Reuses ingestion unchanged; only adds documents on request |
| Connector / ingestion | Yes | New caller of `ingest()`; no change to connectors or the pipeline |
| Auth / document ACLs | No | No auth exists yet; same exposure as `POST /documents` |
| Public API or stream contract | Yes | New `POST /documents/samples` |
| Persistence / migration | No | No schema change; dedupe uses the existing `content_hash` |
| New external dependency or service | No | |

## Impact

- `Dockerfile`, `.dockerignore`
- `backend/src/plumbline/api/app.py`, backend tests
- `docs/architecture/overview.md` or `docs/guides/backend-development-guide.md` (sample path, env var, endpoint)
