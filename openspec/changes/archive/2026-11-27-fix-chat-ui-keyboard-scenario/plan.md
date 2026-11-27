## Task
1.1 Confirm the app matches the delta specs: a keyboard-only pass on the chat (ask, stop, copy, regenerate, open a source) and on the Documents page upload, each step showing a visible focus indicator. Record the result in `verification.md`. If an existing test (e.g. `frontend/src/shell/a11y.test.tsx`) covers the upload control's keyboard reachability, cite it; otherwise add one small test that tabs to the upload input and asserts it receives focus. Check: `npm test` in `frontend/` passes; `openspec validate fix-chat-ui-keyboard-scenario` passes.

## Scope
- Guides read: `docs/guides/frontend-development-guide.md` (sections 4.2 and 5).
- Existing coverage: `a11y.test.tsx` renders the Documents page and checks accessible names, and its outline-none scan includes `DocumentsPage.tsx`. Nothing asserts the upload input is in the tab order or has a focus style. `DocumentsPage.test.tsx` only checks the label and input markup.
- Add one markup test in `DocumentsPage.test.tsx` for the new "Keyboard-only upload" scenario. The frontend has no DOM test environment (no jsdom, no testing-library), so a real "tab to the input" test is not possible without a new dependency. The test asserts what makes the input reachable and visible: a native `<input type="file">`, no `tabindex="-1"`, not `disabled` when idle, and a `focus-visible:outline` class.
- The by-hand pass of the chat controls and the Documents upload goes in `verification.md`. I cannot drive a keyboard from this session, so the by-hand pass needs the user, or the real app via `run`. Until then the result is recorded as "not performed by hand".

## Not in scope
- Any change to app code or styling.
- The guide line `frontend-development-guide.md:265`, which still lists "upload" under keyboard-only use of the chat. It is stale for the same reason as the spec. I will flag it, not edit it (surgical changes).
- Adding jsdom or testing-library.

## Files
| File | Change |
| --- | --- |
| `frontend/src/documents/DocumentsPage.test.tsx` | Add one `it` in the existing `describe`: upload input is native, in the tab order, enabled when idle, has `focus-visible:outline` |
| `openspec/changes/fix-chat-ui-keyboard-scenario/verification.md` | Written later by `rag-verify`, not in this task |

## Tests
- New test in `DocumentsPage.test.tsx` covers the `document-management` scenario "Keyboard-only upload" (reachable, visible focus indicator). The `chat-ui` scenario edit is text-only and covered by `openspec validate`.

## Checks to run
- `cd frontend && npm test`
- `cd frontend && npm run lint` and `npm run typecheck` if those scripts exist (confirm in `package.json`; AGENTS.md lists the exact commands)
- `openspec validate fix-chat-ui-keyboard-scenario`

## Risks / Questions
- The task text says "tabs to the upload input and asserts it receives focus". That is not achievable in this repo's static-markup test setup without a new dependency. The plan substitutes a markup assertion. Approve this, or say if you want jsdom added (a separate decision).
- The by-hand keyboard pass cannot be done by me. Do you want to do it and give me the result, or should I record it as pending?
