## 1. Endpoint

- [ ] 1.1 Add `POST /documents/samples` in `backend/src/plumbline/api/app.py`. It resolves `PLUMBLINE_SAMPLES_DIR`, defaulting to the repo's `evals/connectors/pdf/corpus`, and returns 404 if the directory is missing. It calls `ingest()` on each top-level file and collects `indexed`/`failed`. Tests (API, temp samples dir): all indexed and listed by `GET /documents`; a second call duplicates nothing (`created: false`, document and chunk counts unchanged); a corrupt file lands in `failed` with a reason while the others index; a missing dir returns 404.

## 2. Image

- [ ] 2.1 Ship the samples in the image: a `.dockerignore` re-include for `evals/connectors/pdf/corpus/*.pdf`, `COPY` to `/app/samples`, and `ENV PLUMBLINE_SAMPLES_DIR=/app/samples`. Check: `docker build`, then `docker run --rm <img> sh -c "ls /app/samples; ls /app"` shows the four PDFs and no `evals`/golden/baseline files. Paste the output into verification.

## 3. Docs

- [ ] 3.1 Document the samples path, `PLUMBLINE_SAMPLES_DIR` and `POST /documents/samples` (request, response, 404) in `docs/guides/backend-development-guide.md`, and add a line in `docs/architecture/overview.md` if that page lists endpoints.
