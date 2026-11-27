Issue: #45

## Why

The `chat-ui` "Keyboard-only use" scenario still lists uploading a file, but since #33 upload lives on the Documents page, and no spec covers keyboard-only upload there.

## What Changes

- `openspec/specs/chat-ui/spec.md`, requirement "Accessible, responsive, themed", scenario "Keyboard-only use": the WHEN becomes "they ask a question, stop, copy, regenerate and open a source" (drops "upload a file").
- `openspec/specs/document-management/spec.md`: add requirement "Keyboard-only upload" with the scenario: GIVEN a keyboard-only user on the Documents page, WHEN they upload a file, THEN every step is reachable with a visible focus indicator.
- Out of scope: behaviour or code changes; the "Upload in the new kit" requirement in `chat-ui` stays as is.

## Confirmed Low-Risk

Spec text only: no retrieval, prompts, connectors, ingestion, auth, contract or DB, and it reviews in one sitting.

## Impact

Delta specs in this change, synced to the main specs at archive: 
`openspec/specs/chat-ui/spec.md`, `openspec/specs/document-management/spec.md`.
