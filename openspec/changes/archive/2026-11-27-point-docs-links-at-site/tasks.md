## 1. Docs site links

- [x] 1.1 Define the docs site base URL once and use it in `shell/Sidebar.tsx` and `connectors/ConnectorsPage.tsx`. Point Connector guide at `guides/connector-development-guide/` and How it's measured at `architecture/overview/#evaluate-plumbline-eval`. Remove the now-unused `REPO` constant. Update `ConnectorsPage.test.tsx` and `shell/a11y.test.tsx` to assert the site URLs. Check: `npm test` in `frontend/` passes; `git grep -nE "github.com|\.md['\"]" frontend/src` shows no documentation target (repository links only).
