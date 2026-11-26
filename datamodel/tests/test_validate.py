"""Each validator rule: a broken copy of the sample model -> the error it must raise."""
import pytest

import validate
from model import load
from schema import Column, Question, Rule


def _fk(m, table, ref):
    m.tables[table].columns.append(Column(name="x_id", type="uuid", references=ref, description="x"))


def _rename_table(n):
    def f(m):
        m.tables["organization"].table = "t" * n
    return f


def _rename_column(n):
    def f(m):
        m.tables["org_person"].columns[1].name = "c" * n  # first_name; not in any index or unique
    return f


def _q(status, owner="CTO", answer=None):
    return Question(id="OQ-ORGANIZATION-099", owner=owner, status=status, text="q", answer=answer)


CASES = {
    "fk outside depends_on": (lambda m: _fk(m, "org_clinic_location", "org_person.id"), "does not depend on"),
    "fk not at id": (lambda m: _fk(m, "org_person", "organization.public_id"), "internal id"),
    "table name 64 bytes": (_rename_table(64), "table name is 64 bytes"),
    "column name 64 bytes": (_rename_column(64), f"org_person.{'c' * 64}: column name is 64 bytes"),
    "duplicate prefix": (lambda m: setattr(m.tables["org_person"], "public_id_prefix", "org"), "already used by"),
    "duplicate rule id": (
        lambda m: m.tables["organization"].business_rules.extend([Rule(id="BR-ORGANIZATION-099", text="r")] * 2),
        "already used in",
    ),
    "default on pii column": (lambda m: setattr(m.tables["org_person"].columns[1], "default", "'x'"), "cannot have a default"),
    "sample row unknown column": (lambda m: m.tables["org_person"].sample_rows.append({"nope": 1}), "unknown column nope"),
    "sample row bad enum": (lambda m: m.tables["org_person"].sample_rows.append({"sex_at_birth": "NOPE"}), "not in enum"),
    "sample row ssn": (lambda m: m.tables["org_person"].sample_rows.append({"first_name": "123-45-6789"}), "SSN"),
    "sample row real email": (lambda m: m.tables["org_person"].sample_rows.append({"email": "a@gmail.com"}), "reserved domain"),
    "published with open CMO question": (
        lambda m: m.tables["organization"].open_questions.append(_q("open", owner="CMO")),
        "open CMO question",
    ),
    "answered without answer": (lambda m: m.tables["organization"].open_questions.append(_q("answered")), "has no answer"),
}


def test_valid_model():
    assert validate.validate(load()) == []


@pytest.mark.parametrize("mutate,expected", CASES.values(), ids=CASES.keys())
def test_rule_fails(mutate, expected):
    m = load()
    mutate(m)
    errs = validate.validate(m)
    assert any(expected in e for e in errs), errs


@pytest.mark.parametrize("mutate", [_rename_table(63), _rename_column(63)], ids=["table", "column"])
def test_63_byte_name_has_no_length_error(mutate):
    m = load()
    mutate(m)
    assert not any("name is 63 bytes" in e for e in validate.validate(m))
