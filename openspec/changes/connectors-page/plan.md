## Task
5.1 `builtin` on `GET /connectors` (true when the connector class module is under `plumbline.connectors.builtin`); update `test_connectors_api.py`: built-ins true, a third-party fake false, still exactly four fields

## Scope
- In `backend/src/plumbline/api/app.py`, add one key to each entry of the `/connectors` response: `"builtin": type(c).__module__.startswith("plumbline.connectors.builtin.")`, as in design.md "`builtin` on `GET /connectors`". Name, extensions, contract version and the sort stay as they are.
- `FileConnector` instances come from `registry.load_connectors()`, which instantiates the class from the entry point, so `type(c).__module__` is the class's defining module. Verified for the built-ins: `pyproject.toml` registers `plumbline.connectors.builtin.pdf:PdfConnector` and `...docx:DocxConnector`. For the test fake it is `fake_connectors`, so it is not built in.
- Update `backend/tests/test_connectors_api.py`; this guard is the "still four fields" check.
- Guide read: `docs/guides/backend-development-guide.md` (routes stay thin, 1.1; the response is an allowlist built from named fields, as `/quality` is, 2.5; tests sit with the code, 8).

## Not in scope
- The frontend type and the card's "Built-in" text (6.1, 6.2).
- Guide text for the fourth field (7.1). Until then the backend guide's rule 2.6 says three fields; it is stale for one task, fixed in 7.1.
- A registry change or a hard-coded name list (rejected in design.md).
- Starting or restarting the dev backend; I'll restart it myself only if you ask.

## Files
| File | Change |
| --- | --- |
| `backend/src/plumbline/api/app.py` | add the `builtin` key |
| `backend/tests/test_connectors_api.py` | built-ins true; third-party false; exactly four keys |

## Tests
- Built-ins (spec "Built-in connectors"): the existing test's expected list gains `"builtin": True` for `docx` and `pdf`, which also proves exactly four keys (it compares whole dicts).
- Third-party (spec "Third-party connector"): with the registry patched to `GoodConnector` from `fake_connectors` (`_install`, as the existing failed-connector test does), `/connectors` returns one entry with `builtin` false and exactly `{name, extensions, contract_version, builtin}`.
- The existing failed-connector and name-match tests keep passing unchanged.

## Checks to run
- `cd backend && uv run ruff check . && uv run ruff format --check .`
- `cd backend && uv run python -m pytest -q` (`uv run pytest` fails on this machine with a uv trampoline error; `python -m pytest` works).

## Risks / Questions
- A third-party connector whose package is also named `plumbline.connectors.builtin.*` would read as built-in; that namespace belongs to this repo, so I accept it.
- None otherwise.
