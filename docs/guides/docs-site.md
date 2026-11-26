# Docs site

This site is built from `docs/` with MkDocs and the Material theme, so a
change to code and docs lands in one PR.

```bash
# from the repo root
python -m venv .venv-docs && . .venv-docs/bin/activate   # Scripts\activate on Windows
pip install -r docs-requirements.txt
mkdocs serve                  # live preview on :8000
mkdocs build --strict         # what CI runs; a broken link fails it
```

- Navigation is the explicit `nav` tree in `mkdocs.yml`. A new page needs an entry there.
- Tooling is pinned in `docs-requirements.txt` and is not in the runtime image.
- Netlify builds `main` and posts a deploy preview on each PR (`netlify.toml`).
- Netlify skips a build when a push changes none of `docs/`, `mkdocs.yml`, `docs-requirements.txt` or `netlify.toml`.
