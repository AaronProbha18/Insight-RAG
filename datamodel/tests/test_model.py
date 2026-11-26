import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))

import validate
from conftest import MINI
from model import load
from schema import Check


def test_mini_fixture_is_valid():
    assert validate.validate(load(MINI)) == []


def test_long_constraint_name_rejected():
    m = load(MINI)
    m.tables["note"].checks.append(Check(name="ck_" + "x" * 70, expr="true"))
    assert any("truncates" in e for e in validate.validate(m))


def test_pii_default_rejected():
    m = load(MINI)
    m.tables["account"].columns[1].default = "'x'"
    assert any("cannot have a default" in e for e in validate.validate(m))


def test_unknown_fk_rejected():
    m = load(MINI)
    m.tables["note"].columns[0].references = "nope.id"
    assert any("unknown table" in e for e in validate.validate(m))
