## Task
4.4 Brand assets: move `plumbline-brand/` to `docs/brand/`; add `frontend/public/` with `favicon.svg` (from `plumbline-icon-small.svg`), `favicon.ico`, `apple-touch-icon.png`; add the three icon links to `frontend/index.html`; add the README `<picture>` logo header. Test: a check that every icon link in `index.html` and every logo link in the README resolves to an existing file; `npm run build` output contains the three icons.

## Scope
- Guides read: `docs/guides/frontend-development-guide.md` (assets: nothing restricts a `public/` folder). Vite serves `frontend/public/` at `/` and copies it into `dist/`, which the Dockerfile and the FastAPI static mount (`PLUMBLINE_UI_DIR`) already serve; `vite.config.ts` has no `base`, so root-relative links work.
- Inspected the supplied files: `favicon.ico` holds 16, 32 and 48 px entries; `apple-touch-icon.png` is 180x180; both logo PNGs 1188x240; social preview 1280x640. Nothing to resize or regenerate.
- `mv plumbline-brand docs/brand` (untracked, so no `git mv`). The whole folder is committed, including `social-preview.*`, the 16/32/192/512 PNGs and the source SVGs; none is wired into the app (no web manifest, social preview is the owner step in 5.2).
- `frontend/public/`: copy `plumbline-icon-small.svg` as `favicon.svg`, `favicon.ico`, `apple-touch-icon.png` (copies, so the app does not depend on `docs/`).
- `frontend/index.html` head: the three links from the brand README (`/favicon.svg` svg, `/favicon.ico` sizes 48x48, `/apple-touch-icon.png`).
- `README.md`: replace the `# Plumbline` line with `<h1>` wrapping the `<picture>` (dark source `docs/brand/plumbline-logo-dark.svg`, light `plumbline-logo.svg`, `alt="Plumbline"`, `height="64"`), so the page keeps one heading and no duplicate wordmark.
- `docs/brand/README.md`: replace the "Suggested placement (as part of the rename change)" section with a short statement of what is placed where; keep the asset table, colours and font licence.
- `AGENTS.md` layout block: add a `docs/brand/` line and correct `reports/ (gitignored)` to say baselines are force-added.
- New check `backend/tests/test_brand_assets.py` (see Tests).

## Not in scope
- A web app manifest, using the 192/512 icons in the app, a logo in the app header.
- Uploading the social preview (5.2, owner).
- Changing the brand files themselves.

## Files
| File | Change |
| --- | --- |
| `plumbline-brand/**` -> `docs/brand/**` | move (15 assets plus README) |
| `docs/brand/README.md` | placement section rewritten |
| `frontend/public/{favicon.svg,favicon.ico,apple-touch-icon.png}` | new (copies) |
| `frontend/index.html` | three `<link>` tags |
| `README.md` | logo header |
| `AGENTS.md` | layout lines |
| `backend/tests/test_brand_assets.py` | new, about 25 lines |

## Tests
- Spec "Brand assets ship with the product", scenario "Built UI carries the icons": `test_brand_assets.py` reads `frontend/index.html`, takes every `<link ... href="/x">` for an icon and asserts `frontend/public/x` exists.
- Scenario "README logo resolves": the same file reads `README.md`, takes every `src=`/`srcset=` in the header `<picture>` and asserts the path exists relative to the repo root.
- Build check (manual, recorded in the report): `npm run build`, then `dist/` contains `favicon.svg`, `favicon.ico`, `apple-touch-icon.png` and `dist/index.html` links them.
- Serve check (one-off, not a test): with `PLUMBLINE_UI_DIR=frontend/dist`, `GET /favicon.svg`, `/favicon.ico`, `/apple-touch-icon.png` return 200 through the real FastAPI mount.
- `test_old_names.py` still passes (brand README has no old names; verified by running it with the folder untracked earlier).

## Checks to run
```bash
mv plumbline-brand docs/brand
mkdir frontend/public && cp docs/brand/plumbline-icon-small.svg frontend/public/favicon.svg && cp docs/brand/favicon.ico docs/brand/apple-touch-icon.png frontend/public/
cd frontend && npm run lint && npm run build && npm test
ls dist | grep -E "favicon|apple-touch"; grep -n "<link" dist/index.html
cd ../backend && uv run pytest tests/test_brand_assets.py tests/test_old_names.py -q
PLUMBLINE_UI_DIR=../frontend/dist uv run python -c "from fastapi.testclient import TestClient; import plumbline.api.app as a; c=TestClient(a.app); print([c.get(p).status_code for p in ('/favicon.svg','/favicon.ico','/apple-touch-icon.png')])"
uv run pytest -q && uv run ruff check . && uv run ruff format --check .
cd .. && python scripts/check_links.py && openspec validate --all
```

## Risks / Questions
- **README header form**: default replaces the markdown title with `<h1><picture>...` (one heading, no duplicate wordmark). Alternative: keep `# Plumbline` and put the picture above it, which shows the name twice. GitHub renders `picture`/`source` inside README HTML (the brand README's own snippet relies on it); I cannot render GitHub here, so the light/dark switch is UNVERIFIED until the PR page is viewed.
- **Heading anchor**: `docs/architecture/overview.md` links `README.md#quickstart-docker`, not the title anchor, so the h1 change breaks no link (`check_links.py` confirms).
- **Copies vs references**: the three icons exist twice (`docs/brand/` source, `frontend/public/` served). Intentional so the frontend build stays self-contained; the test would catch a deleted copy, not a divergent one.
- **Browser caching/favicons** are visual and cannot be checked headlessly here; the HTTP 200 check covers wiring only.
