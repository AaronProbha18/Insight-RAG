"""DDL generation and the migrate CLI, without Postgres."""
import json

import ddl
from conftest import run_tool
from model import load

SYSTEM = ("id", "public_id", "created_at", "updated_at")


def test_system_columns_injected():
    m = load()
    assert not any(c.name in SYSTEM for t in m.tables.values() for c in t.columns), "system column written in YAML"
    t = ddl.build_objects(m)["tables"]
    org = t["organization"]
    cols = list(org["columns"])
    assert cols[:2] == ["id", "public_id"] and cols[-2:] == ["created_at", "updated_at"]
    assert org["columns"]["id"]["def"] == "id uuid NOT NULL DEFAULT gen_random_uuid()"
    assert org["constraints"]["pk_organization"] == "PRIMARY KEY (id)"
    assert org["constraints"]["uq_organization_public_id"] == "UNIQUE (public_id)"
    assert org["constraints"]["ck_organization_public_id"] == "CHECK (public_id ~ '^org_[0-9A-Za-z]+$')"
    assert "updated_at" not in t["person_consent_grant"]["columns"]  # fully immutable
    assert "updated_at" in t["membership_order"]["columns"]  # immutable with mutable_columns


def test_enum_is_check_not_type():
    objs = ddl.build_objects(load())
    assert objs["tables"]["organization"]["constraints"]["ck_organization_status_enum"] == \
        "CHECK (status IN ('ACTIVE', 'INACTIVE'))"
    assert "CREATE TYPE" not in ddl.render_full(objs)


def _edit(path, old, new):
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))


def test_migration_written(workcopy):
    _edit(workcopy / "domains/organization/tables/organization.yaml", "  - {name: effective_from,",
          "  - {name: website, type: text, nullable: true, description: Public website.}\n  - {name: effective_from,")
    r = run_tool(workcopy, "migrate.py", "--name", "add_x")
    assert r.returncode == 0, r.stdout + r.stderr
    sql = (workcopy / "migrations/0002_add_x.sql").read_text()
    assert "BEGIN;" in sql and "ALTER TABLE organization ADD COLUMN website text;" in sql and sql.endswith("COMMIT;\n")
    lock = json.loads((workcopy / "model.lock.json").read_text())
    assert "website" in lock["tables"]["organization"]["columns"]
    assert run_tool(workcopy, "migrate.py", "--dry-run").stdout.startswith("no changes")


def test_destructive_change_refused_then_approved(workcopy):
    _edit(workcopy / "domains/organization/tables/org_clinic_location.yaml",
          "  - {name: phone, type: text, nullable: true, pii: indirect, description: Site phone number.}\n", "")
    lock_before = (workcopy / "model.lock.json").read_text()
    r = run_tool(workcopy, "migrate.py", "--name", "drop_x")
    assert r.returncode == 2 and "REFUSED" in r.stderr
    assert not (workcopy / "migrations/0002_drop_x.sql").exists()
    assert (workcopy / "model.lock.json").read_text() == lock_before

    r = run_tool(workcopy, "migrate.py", "--name", "drop_x", "--allow-destructive")
    assert r.returncode == 0, r.stdout + r.stderr
    sql = (workcopy / "migrations/0002_drop_x.sql").read_text()
    assert "-- RISKY, approved with --allow-destructive:" in sql
    assert "ALTER TABLE org_clinic_location DROP COLUMN phone;  -- DESTRUCTIVE" in sql
