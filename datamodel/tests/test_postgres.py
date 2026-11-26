"""Generated SQL against a real Postgres, through the psql CLI.

Set DATAMODEL_TEST_DATABASE_URL to a maintenance database (e.g. .../postgres);
each test creates and drops its own temporary databases. Skipped otherwise.
test_drift.py imports the helpers below.
"""
import os
import shutil
import subprocess
import uuid
from urllib.parse import urlsplit, urlunsplit

import pytest

import ddl
from conftest import MINI
from model import load

URL = os.environ.get("DATAMODEL_TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.pg,
    pytest.mark.skipif(not URL, reason="DATAMODEL_TEST_DATABASE_URL not set"),
    pytest.mark.skipif(not shutil.which("psql"), reason="psql not on PATH"),
]


def _db_url(name):
    u = urlsplit(URL)
    return urlunsplit(u._replace(path="/" + name))


def psql(url, sql, ok=True):
    r = subprocess.run(["psql", url, "-X", "-q", "-At", "-v", "ON_ERROR_STOP=1", "-c", sql],
                       capture_output=True, text=True)
    if ok:
        assert r.returncode == 0, r.stderr
    return r


def apply(url, path):
    r = subprocess.run(["psql", url, "-X", "-q", "-v", "ON_ERROR_STOP=1", "-f", str(path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"{path.name}: {r.stderr}"


@pytest.fixture
def newdb():
    made = []

    def make():
        name = f"dm_test_{uuid.uuid4().hex[:10]}"
        psql(URL, f"CREATE DATABASE {name}")
        made.append(name)
        return _db_url(name)

    yield make
    for name in made:
        psql(URL, f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


def test_fixture_ddl_applies(newdb, tmp_path):
    f = tmp_path / "schema.sql"
    f.write_text(ddl.render_full(ddl.build_objects(load(MINI))))
    url = newdb()
    apply(url, f)
    tables = psql(url, "SELECT string_agg(tablename, ',' ORDER BY tablename) FROM pg_tables "
                       "WHERE schemaname = 'public'").stdout.strip()
    assert tables == "account,event,note"
