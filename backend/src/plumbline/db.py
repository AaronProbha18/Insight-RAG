"""Database location, shared by the API, migrations, eval runner and tests."""

import os

# Matches the docker-compose `db` service defaults; local development only.
DEFAULT_DATABASE_URL = "postgresql://plumbline:plumbline@localhost:5432/plumbline"


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
