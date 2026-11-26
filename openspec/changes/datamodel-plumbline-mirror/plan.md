## Task
4.2 `plumbline-drift` job in `.github/workflows/datamodel.yml` (pgvector `pg18` service, `PLUMBLINE_EMBEDDING_DIM=8`, paths add `backend/migrations/**`): alembic head into one DB, generated DDL into another, `drift.py`. Not a required check. Test: green on the PR; a scratch commit making one YAML column nullable fails the job, naming the column (then reverted).

## Scope
- `.github/workflows/datamodel.yml`:
  - `paths` (push and pull_request) gain `backend/migrations/**`.
  - New job `plumbline-drift` (`runs-on: ubuntu-latest`) next to the existing `datamodel` job, which is untouched.
  - Service `postgres`: `pgvector/pgvector:pg18` (same image as `ci.yml`'s backend job), `POSTGRES_PASSWORD` a throwaway CI credential with the same comment the existing job uses, `pg_isready` health check.
  - Job `env`: `PLUMBLINE_EMBEDDING_DIM: 8`, and `ADMIN_URL`, `ALEMBIC_URL`, `GENERATED_URL` pointing at the service's `postgres`, `alembic` and `generated` databases.
  - Steps (`actions/checkout@v7`, `astral-sh/setup-uv@v7`, as the other jobs):
    1. `psql "$ADMIN_URL" -v ON_ERROR_STOP=1 -c "CREATE DATABASE alembic"` and the same for `generated`.
    2. Alembic head, `working-directory: backend`: `uv sync --locked`, then `DATABASE_URL=$ALEMBIC_URL uv run alembic upgrade head`. No `--extra ollama`: `migrations/env.py` imports only `plumbline.config` and `plumbline.db`.
    3. Generated schema, `working-directory: datamodel`: `uv sync --locked`, `make ddl MODEL=models/plumbline`, then `psql "$GENERATED_URL" -v ON_ERROR_STOP=1 -c "CREATE EXTENSION vector" -f models/plumbline/build/schema.sql`.
    4. `uv run python tools/drift.py "$ALEMBIC_URL" "$GENERATED_URL"` (in `datamodel`).
  - The job isn't added to branch protection. The workflow's existing comment about path-filtered checks not being required already covers it; I'll extend it to name this job. No repository settings are changed.
- If the first real run shows genuine drift between `models/plumbline` and Alembic (for example a check expression Postgres renders differently), fixing the YAML in `models/plumbline/` is in scope here: design.md says those are "decided per column ... by running the gate". Alembic stays untouched.
- Guides read: `docs/guides/backend-development-guide.md` isn't needed (no `backend/` file changes; the job only runs Alembic as `Dockerfile` already does). I read `ci.yml` for the service and env conventions, `backend/migrations/env.py` and `backend/src/plumbline/db.py` for `DATABASE_URL`.

## Not in scope
- Docs, the ADR and AGENTS.md (5.1).
- Required-check or branch-protection settings.
- Any change to `backend/`.

## Files
| File | Change |
| --- | --- |
| `.github/workflows/datamodel.yml` | `backend/migrations/**` in both `paths`; new `plumbline-drift` job; comment names it |
| `datamodel/models/plumbline/**` | only if the gate finds real drift (see Scope) |

## Tests
The task's tests need CI, and I can't push (the loop never pushes). So:
- **Before commit, locally, with the job's exact steps:** a throwaway `pgvector/pgvector:pg18` container, as in 4.1, with the repo's `backend/` and `datamodel/` copied in and Python from the `uv` image:
  - **Positive:** Alembic head into `alembic`, generated DDL into `generated`, `drift.py` exits 0 with "no drift". This is the first real comparison of Alembic with `models/plumbline`.
  - **Negative:** in a scratch copy only (never the repo), make one YAML column nullable (e.g. `feedback.rating`), regenerate and re-run: exit 1, output names `feedback` and `rating`.
- **YAML:** parse the workflow with Python's `yaml` and check the job's keys. `actionlint` isn't installed here, so that's a syntax check only.
- **After you push (the PR's CI run):** the green `plumbline-drift` job is the first half of the task's test. The scratch-commit half (a nullable YAML column fails the job, then reverted) needs a push too. I'll leave it as an explicit pending item in `verification.md`, and do it only if you approve the pushes.

## Checks to run
- The local container run above, positive and negative.
- `cd datamodel && uv run python -m pytest -q -rs tests` (unchanged, to prove nothing else moved).
- `python -c "import yaml; yaml.safe_load(open('.github/workflows/datamodel.yml'))"`.

## Risks / Questions
- `paths` is workflow-level, so the existing `datamodel` job now also runs on PRs that touch only `backend/migrations/**`. It's quick and harmless; the alternative (a separate workflow file) contradicts "a second job in `datamodel.yml`" from triage.
- UNVERIFIED until the local run: that `alembic upgrade head` works with only `uv sync --locked` (no `--extra ollama`), and that `make ddl MODEL=models/plumbline` works on the runner (`make` is available there, as the existing job already relies on it).
- Can't be verified before a push: that the job is green on GitHub's runner. Local container results are strong evidence, but CI is the authority.
