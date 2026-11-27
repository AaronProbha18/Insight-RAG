## 1. Spec alignment

- [ ] 1.1 Confirm the app matches the delta specs: a keyboard-only pass on the chat (ask, stop, copy, regenerate, open a source) and on the Documents page upload, each step showing a visible focus indicator. Record the result in `verification.md`. If an existing test (e.g. `frontend/src/shell/a11y.test.tsx`) covers the upload control's keyboard reachability, cite it; otherwise add one small test that tabs to the upload input and asserts it receives focus. Check: `npm test` in `frontend/` passes; `openspec validate fix-chat-ui-keyboard-scenario` passes.
