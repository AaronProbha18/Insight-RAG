"""Generated SQL against a real Postgres 16, through the psql and pg_dump CLIs.

Set DATAMODEL_TEST_DATABASE_URL to a maintenance database (e.g. .../postgres);
each test creates and drops its own temporary databases. Skipped otherwise.
"""
import difflib
import os
import shutil
import subprocess
import sys
import uuid
from itertools import groupby
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest

ROOT = Path(__file__).resolve().parent.parent
URL = os.environ.get("DATAMODEL_TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.pg,
    pytest.mark.skipif(not URL, reason="DATAMODEL_TEST_DATABASE_URL not set"),
    pytest.mark.skipif(not (shutil.which("psql") and shutil.which("pg_dump")), reason="psql/pg_dump not on PATH"),
]

# Synthetic rows only.
ORG = "INSERT INTO organization (public_id, display_name, legal_name, effective_from) VALUES ('org_t1', 'Example', 'Example LLC', '2026-01-01')"
PERSON = ("INSERT INTO org_person (public_id, organization_id, first_name, last_name, email) "
          "SELECT 'per_t1', id, 'Test', 'Member', 't@example.invalid' FROM organization")
PLAN = ("INSERT INTO org_membership_plan (public_id, organization_id, plan_name, annual_price) "
        "SELECT 'plan_t1', id, 'Essential', 1200 FROM organization")
ORDER = ("INSERT INTO membership_order (public_id, person_id, selected_plan_id, billing_frequency, selected_plan_amount) "
         "SELECT 'ord_t1', p.id, pl.id, 'YEARLY', 1200 FROM org_person p, org_membership_plan pl")
GRANT = ("INSERT INTO person_consent_grant (public_id, person_id, policy_code, decision, signed_at) "
         "SELECT 'cg_t1', id, 'TERMS', 'GRANTED', now() FROM org_person")


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


def migrated(newdb, root=ROOT):
    url = newdb()
    for f in sorted((root / "migrations").glob("[0-9][0-9][0-9][0-9]_*.sql")):
        apply(url, f)
    return url


def test_immutable_delete_and_update_rejected(newdb):
    url = migrated(newdb)
    for s in (ORG, PERSON, GRANT):
        psql(url, s)
    r = psql(url, "DELETE FROM person_consent_grant", ok=False)
    assert r.returncode != 0 and "immutable" in r.stderr
    r = psql(url, "UPDATE person_consent_grant SET policy_code = 'OTHER'", ok=False)
    assert r.returncode != 0 and "immutable" in r.stderr


def test_allow_listed_update_advances_updated_at(newdb):
    url = migrated(newdb)
    for s in (ORG, PERSON, PLAN, ORDER):
        psql(url, s)
    r = psql(url, "UPDATE membership_order SET selected_plan_amount = 1", ok=False)
    assert r.returncode != 0 and "immutable" in r.stderr
    before = psql(url, "SELECT updated_at FROM membership_order").stdout.strip()
    psql(url, "UPDATE membership_order SET order_status = 'PAID'")  # own transaction, so now() moves
    after = psql(url, "SELECT updated_at > %s::timestamptz, order_status FROM membership_order"
                 % f"'{before}'").stdout.strip()
    assert after == "t|PAID"


def _normalise(dump):
    """Drop comments/SET/\\restrict noise; sort column lines and column comments, whose order follows column order."""
    lines = [ln for ln in dump.splitlines() if ln and not ln.startswith(("--", "SET ", "SELECT pg_catalog.", "\\restrict", "\\unrestrict"))]
    out, cols = [], None
    for ln in lines:
        if cols is None:
            out.append(ln)
            if ln.startswith("CREATE TABLE"):
                cols = []
        elif ln.startswith(");"):
            out += sorted(c.rstrip(",") for c in cols) + [ln]
            cols = None
        else:
            cols.append(ln)
    # Column comments also follow column order.
    return "\n".join(
        "\n".join(sorted(g)) if is_comment else "\n".join(g)
        for is_comment, g in ((k, list(g)) for k, g in groupby(out, lambda ln: ln.startswith("COMMENT ON COLUMN")))
    )


def _dump(url):
    r = subprocess.run(["pg_dump", "-s", "--no-owner", "--no-privileges", url], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return _normalise(r.stdout)


def test_migrations_equal_fresh_build(newdb, tmp_path):
    work = tmp_path / "dm"
    for d in ("domains", "enums", "tools", "migrations"):
        shutil.copytree(ROOT / d, work / d, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "model.lock.json", work)

    # A YAML change: a nullable column (ADD COLUMN) and an enum value (constraint drop + add).
    org = work / "domains/organization/tables/organization.yaml"
    text = org.read_text()
    text = text.replace(
        "  - {name: effective_from,",
        "  - {name: website, type: text, nullable: true, description: Public website.}\n  - {name: effective_from,", 1)
    org.write_text(text)
    enum = work / "enums/order_status.yaml"
    enum.write_text(enum.read_text().replace("EXPIRED]", "EXPIRED, REFUNDED]"))

    env = {**os.environ, "PYTHONPATH": str(work / "tools")}
    for args in (["migrate.py", "--name", "test_change"], ["ddl.py"]):
        r = subprocess.run([sys.executable, str(work / "tools" / args[0]), *args[1:]],
                           capture_output=True, text=True, env=env, cwd=work)
        assert r.returncode == 0, r.stdout + r.stderr
    assert (work / "migrations/0002_test_change.sql").exists()

    incremental = migrated(newdb, work)
    fresh = newdb()
    apply(fresh, work / "build/schema.sql")
    a, b = _dump(incremental), _dump(fresh)
    assert a == b, "\n".join(difflib.unified_diff(a.splitlines(), b.splitlines(), "incremental", "fresh", lineterm=""))
