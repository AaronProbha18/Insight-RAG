# ADR 0004: Rename the product to Plumbline

## Status

Proposed. Introduced by the `rename-to-plumbline` change (issue #6).

## Context

The project shipped as `insight-rag`. That name leaks into the Python
package, the distribution, env vars, the connector entry-point group, the
eval command, Docker names and the default database credentials. The
product is now Plumbline.

## Decision

| Thing | Old | New |
| --- | --- | --- |
| Python package | `insight_rag` | `plumbline` |
| Distribution | `insight-rag` | `plumbline-rag` (bare `plumbline` is taken on PyPI; `plumbline-rag` was free on 2026-11-23) |
| Eval command | `insight-rag-eval` | `plumbline-eval` |
| Env vars | `INSIGHT_RAG_*` | `PLUMBLINE_*` (same suffixes) |
| Connector entry-point group | `insight_rag.connectors` | `plumbline.connectors` |
| Default Postgres user/password/db | `insight_rag` | `plumbline` |

- **No fallback.** Old names stop working at once. If any `INSIGHT_RAG_*`
  variable is set, startup fails with a message naming each replacement.
  Connectors registered only under `insight_rag.connectors` are not loaded;
  the registry logs a warning naming `plumbline.connectors`. These two
  checks are the only code that knows the old names, and they stay until
  a later ADR removes them.
- **Database.** No data migration. Installs that want to keep an existing
  volume set `POSTGRES_USER`, `POSTGRES_PASSWORD` and `POSTGRES_DB` (or
  `DATABASE_URL`) to the old values; see `docs/migration/plumbline.md`.
- **Repository.** The GitHub repo is renamed and its description replaced
  as part of the change; GitHub redirects the old URL.

## Alternatives

- **Read old names with a deprecation warning for one release.** Kinder
  to existing setups, but keeps two names alive and needs precedence rules.
  Rejected by the product owner: there are few installs to break.
- **Keep the old database defaults.** Zero upgrade friction, but leaves the
  old name in every compose file and connection string.

## Consequences

- Existing `.env` files and third-party connectors break until edited; the
  error and warning say exactly what to change.
- Old names may appear only in this ADR, the migration notes, the two
  detection checks, and historical `openspec/changes/**` records.
