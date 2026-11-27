## Task
1.6 Docs: describe the first-run screen and the sample action in `docs/guides/frontend-development-guide.md`; `mkdocs build --strict` passes.

## Scope
- Guides read: `docs/guides/frontend-development-guide.md`, `docs/guides/docs-site.md`, `docs/architecture/overview.md:62` and `backend-development-guide.md:131` (the samples endpoint, for accuracy).
- Edit `frontend-development-guide.md` in four places, matching what the code does now:
  1. §1.1 file tree: add `chat/FirstRun.tsx` (first-run screen and the sample action); update the `useDocuments.ts` and `documents/api.ts` lines (`loaded`, `trySamples`; `POST /documents/samples`).
  2. §2.1: `AppShell` shows nothing on the chat page until the list loads, shows `FirstRun` for zero documents (not on a failed load), and keeps `ChatPane` mounted but hidden meanwhile.
  3. §2.6: after the samples are indexed `ChatPane` puts one sample question first (`samples` prop).
  4. §5 accessibility: a first-run bullet (`<h1>`, Upload is a real link, Try is a native button, errors in `role="status"`, *Checked by* `shell/a11y.test.tsx` and `chat/FirstRun.test.tsx`); §7.1 manual-check list: Tab order and focus rings on the first-run screen, and the sample flow (documents appear, first-run gives way to the chat with the sample question).
- Verify `mkdocs build --strict`: mkdocs is not installed here, so create a throwaway venv in the scratchpad (outside the repo) and `pip install -r docs-requirements.txt` (pinned `mkdocs==1.6.1`, `mkdocs-material==9.8.2`, per `docs-site.md`). This is a network install of those two pinned packages; say if you would rather skip it and rely on CI.

## Not in scope
- `docs/architecture/overview.md` (no module boundary or endpoint changed; the existing diagram already lists `POST /documents/samples`), the backend guide, README, `mkdocs.yml` (no new page), any code.

## Files
| File | Change |
| --- | --- |
| `docs/guides/frontend-development-guide.md` | the four edits above |

## Tests
- `mkdocs build --strict` passes (broken links fail it). Every file or test name written into the guide must exist: checked by grep.

## Checks to run
- `<scratch venv>/Scripts/python -m mkdocs build --strict -d <scratch dir>`; `git status` shows only the guide changed.

## Risks / Questions
- The manual browser checks I add to the guide have not been run by me; the guide will list them as "check by hand", as it does for the other pages.
