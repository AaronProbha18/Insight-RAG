Issue: #59

## Why

The Connectors page links to Markdown files on GitHub although the docs site is deployed at https://plumbline-docs.netlify.app/ (#48), and the site base URL is repeated per file.

## What Changes

- Define the docs site base URL once (`https://plumbline-docs.netlify.app/`) and use it in the sidebar's "Docs and help" link and on the Connectors page.
- Connector guide → `<base>guides/connector-development-guide/`.
- How it's measured → `<base>architecture/overview/#evaluate-plumbline-eval`: the overview's existing "Evaluate" section. No new docs page (triage chose this over adding one or keeping the GitHub link).
- Update `connectors/ConnectorsPage.test.tsx` and `shell/a11y.test.tsx` (it asserts both external links start with `github.com/`).
- Out of scope: docs content, the docs site, new links.

## Confirmed Low-Risk

Two link targets and one shared constant in the frontend: no retrieval, prompts, connectors, ingestion, auth, contract or DB. Verified on the live site: pages use a trailing slash (`.md` is 404), and the `#evaluate-plumbline-eval` anchor exists.

## Impact

`frontend/src/shell/Sidebar.tsx`, `frontend/src/connectors/ConnectorsPage.tsx`, their tests, `frontend/src/shell/a11y.test.tsx`.
