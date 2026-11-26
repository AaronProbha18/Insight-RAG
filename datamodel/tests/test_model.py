import copy
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))

import ddl
import migrate
import validate
from model import load


def test_sample_model_is_valid():
    assert validate.validate(load()) == []


def test_immutable_trigger_generated():
    objs = ddl.build_objects(load())
    assert "trg_person_consent_grant_immutable" in objs["tables"]["person_consent_grant"]["triggers"]
    assert "tg_person_consent_grant_updated_at" not in objs["tables"]["person_consent_grant"]["triggers"]
    # partial case keeps updated_at and allows order_status
    t = objs["tables"]["membership_order"]["triggers"]["trg_membership_order_immutable"]
    assert "'order_status', 'updated_at'" in t


def test_long_constraint_name_rejected():
    m = load()
    t = m.tables["org_person"]
    t.checks.append(type(t.checks[0] if t.checks else m.tables["organization"].checks[0])(name="ck_" + "x" * 70, expr="true"))
    assert any("truncates" in e for e in validate.validate(m))


def test_pii_default_rejected():
    m = load()
    m.tables["org_person"].columns[1].default = "'x'"
    assert any("cannot have a default" in e for e in validate.validate(m))


def test_unknown_fk_rejected():
    m = load()
    m.tables["org_person"].columns[0].references = "nope.id"
    assert any("unknown table" in e for e in validate.validate(m))


def test_diff_safe_and_risky():
    old = ddl.build_objects(load())
    new = copy.deepcopy(old)
    new["tables"]["organization"]["columns"]["note"] = {"def": "note text", "comment": "x"}
    stmts, bad = migrate.diff(old, new)
    assert any("ADD COLUMN note text" in s for s in stmts) and not bad
    new["tables"]["organization"]["columns"]["req"] = {"def": "req int NOT NULL", "comment": "x"}
    _, bad = migrate.diff(old, new)
    assert bad


def test_lock_matches_model():
    import json
    lock = json.loads((pathlib.Path(__file__).resolve().parent.parent / "models/sample/model.lock.json").read_text())
    stmts, bad = migrate.diff(lock, ddl.build_objects(load()))
    assert stmts == [] and bad == []
