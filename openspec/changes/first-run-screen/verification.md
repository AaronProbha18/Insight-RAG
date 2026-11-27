## Checks Run

Run from `D:\PROJECTS\Plumbline\frontend` (upper-case drive, guide 7.4) on branch `feat/35-first-run-screen`, all in this session:

- `npm test` -> 28 files, 193 tests passed.
- `npm run lint` -> oxlint, no warnings.
- `npm run build` -> `tsc -b` and Vite build succeeded.
- `openspec validate --all` -> 15 passed, 0 failed (includes `first-run-screen`).
- `mkdocs build --strict` (pinned `mkdocs==1.6.1`, `mkdocs-material==9.8.2`, scratch venv) -> built, no warnings.
- `backend/.venv python -m pytest tests/test_documents_api.py -k samples` -> 4 passed (includes `test_samples_idempotent`, which the "run twice, list unchanged" case relies on). `uv run` failed here ("trampoline failed to canonicalize script path"), so I ran the venv's python directly.

Not run: `answer-quality-change` eval and the datamodel drift check (the issue has neither label; no backend, schema or migration change).

## Confirms

This tier has no specs; each tasks.md test maps as:

- 1.1 `loaded` -> `loadDocuments` 200/500/offline and first-render `loaded` false in `useDocuments.test.tsx`.
- 1.2 `indexSamples` -> parse, POST, 404/500/offline throw in `api.test.ts`.
- 1.3 `FirstRun` -> Upload link and plain-click, Try handler and disabled state, error text, `trySamples` outcomes in `FirstRun.test.tsx` and `useDocuments.test.tsx`.
- 1.4 `AppShell` gating -> zero documents shows first-run and "Documents (0)", not loaded shows nothing, documents present or all deleted, load error shows chat, in `AppShell.test.tsx`; sample suggestion in `App.test.tsx`; run-twice idempotence in the backend test above.
- 1.5 accessibility -> `shell/a11y.test.tsx` first-run block; contrast uses pairs already in `design-tokens.test.ts`.
- 1.6 docs -> strict build above.

**Checked by hand (by the author, in a browser, against the local stack with the `plumbline-ollama` container):** the first-run screen on an empty list; its match to the `First run (no documents)` page of the Plumbline App Shell wireframe (task 1.7); the failure message when the samples can't be indexed (Ollama stopped); and the samples flow with Ollama running (first-run gives way to the chat). The author reported "LGTM". Not recorded: light versus dark theme, Tab order and focus rings, and deleting every document to bring first-run back; those stay open from the frontend guide's manual list.

Automated checks re-run after task 1.7: `npm test` 28 files / 193 tests, `npm run lint`, `npm run build` all pass.

## Deviation

- `trySamples` lives in `documents/useDocuments.ts`, not `chat/FirstRun.tsx` (avoids the `only-export-components` lint warning).
- `App.test.tsx` now renders `ChatPane` instead of `<App />`, and `AppShell.test.tsx`/`a11y.test.tsx` fixtures gained `loaded`.
- Task 1.7 (added after review): the screen follows the wireframe's layout, type scale, spacing, copy and step cards, with the app's own colour tokens instead of the wireframe's hex values.
- The proposal says the sample question comes "from `returns-policy.md`"; the shipped sample is `returns-policy.pdf`, so the question is worded without a filename.
- If the list refetch after indexing fails, the screen stays on first-run with no message.

**Verdict: PASS**, with the light/dark, keyboard and delete-all manual checks still open.
