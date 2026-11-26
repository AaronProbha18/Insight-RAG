"""Each validator rule: a broken copy of the mini fixture model -> the error it must raise."""
import pytest

import validate
from conftest import MINI
from model import load
from schema import Column, Question


def _fk(m, table, ref):
    m.tables[table].columns.append(Column(name="x_id", type="bigint", references=ref, description="x"))


def _rename_table(n):
    def f(m):
        m.tables["account"].table = "t" * n
    return f


def _rename_column(n):
    def f(m):
        m.tables["account"].columns[1].name = "c" * n  # first_name; not in any index or unique
    return f


def _q(status, owner="CTO", answer=None):
    return Question(id="OQ-ACCOUNT-099", owner=owner, status=status, text="q", answer=answer)


CASES = {
    "fk outside depends_on": (lambda m: _fk(m, "event", "account.id"), "does not depend on"),
    "fk not at id": (lambda m: _fk(m, "note", "account.public_id"), "internal id"),
    "table name 64 bytes": (_rename_table(64), "table name is 64 bytes"),
    "column name 64 bytes": (_rename_column(64), f"account.{'c' * 64}: column name is 64 bytes"),
    "duplicate prefix": (lambda m: setattr(m.tables["note"], "public_id_prefix", "acc"), "already used by"),
    "default on pii column": (lambda m: setattr(m.tables["account"].columns[1], "default", "'x'"), "cannot have a default"),
    "published with open CMO question": (
        lambda m: m.tables["account"].open_questions.append(_q("open", owner="CMO")),
        "open CMO question",
    ),
    "answered without answer": (lambda m: m.tables["account"].open_questions.append(_q("answered")), "has no answer"),
    "duplicate question id": (
        lambda m: m.tables["account"].open_questions.extend([_q("open"), _q("open")]),
        "OQ-ACCOUNT-099 already used in",
    ),
}


def test_valid_model():
    assert validate.validate(load(MINI)) == []


@pytest.mark.parametrize("mutate,expected", CASES.values(), ids=CASES.keys())
def test_rule_fails(mutate, expected):
    m = load(MINI)
    mutate(m)
    errs = validate.validate(m)
    assert any(expected in e for e in errs), errs


@pytest.mark.parametrize("mutate", [_rename_table(63), _rename_column(63)], ids=["table", "column"])
def test_63_byte_name_has_no_length_error(mutate):
    m = load(MINI)
    mutate(m)
    assert not any("name is 63 bytes" in e for e in validate.validate(m))
