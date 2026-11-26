## Task
8.1 `backend/tests/test_ui_mount.py`: with the UI directory set, `GET /app/connectors` returns `index.html`, and `GET /connectors` returns the API's JSON, not `index.html` (app-shell scenarios "Shared link in production" and "API paths are not affected")

## Scope
- Test-only change in `backend/tests/test_ui_mount.py`, using its existing `_client(monkeypatch, ui_dir)` helper (sets `PLUMBLINE_UI_DIR`, reloads the app module).
- Extend `test_app_path_serves_index`: request `/app/connectors` as well as `/app/documents`, each returning 200 with the `index.html` text. (Same test, a loop over the two paths, so the Connectors URL is named.)
- Extend `test_api_and_unknown_paths_are_not_index`: with the same client, `GET /connectors` returns the API list (a JSON list of loaded connectors, each with `name`), and the body has no `<title>`. It needs no database: `/connectors` reads the registry only, so no `get_conn` override is required for it; the test already overrides `get_conn` for `/documents`.
- No production code changes. If a new assertion fails on real behaviour, I stop and report it rather than change the route.
- Guide read: `docs/guides/backend-development-guide.md` (8 testing; the UI mount rules in `docs/guides/frontend-development-guide.md` §1.4 name this test file as the enforcer).

## Not in scope
- Any change to `api/app.py`.
- Other scenarios (all other rows of `verification.md` are covered).
- Re-running `rag-verify` (the step after this task).

## Files
| File | Change |
| --- | --- |
| `backend/tests/test_ui_mount.py` | two existing tests extended |

## Tests
- "Shared link in production" (spec app-shell): `GET /app/connectors` with `PLUMBLINE_UI_DIR` set returns 200 and the `index.html` content; `/app/documents` still does.
- "API paths are not affected" (spec app-shell): `GET /connectors` returns the JSON list (contains the built-in `pdf` and `docx`), not `index.html`; `GET /documents` and an unknown path behave as before.

## Checks to run
- `cd backend && uv run ruff check . && uv run ruff format --check .`
- `cd backend && uv run python -m pytest tests/test_ui_mount.py -q`, then `uv run python -m pytest -q` (`uv run pytest` fails here with a uv trampoline error)

## Risks / Questions
- `_client` reloads `plumbline.api.app`, which resets the `@cache` on `get_connectors`; `/connectors` then loads the real built-in entry points through `load_connectors()`, which depends on the package being installed in the venv (it is: the existing `test_connectors_api` relies on the same). None otherwise.
