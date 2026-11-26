"""The toolkit does only what Plumbline needs: no migration generation, enums, immutable tables or sample rows."""
import re

import pytest
from pydantic import ValidationError

import explorer
from conftest import MINI, ROOT
from model import load


def test_migration_tooling_absent():
    assert not (ROOT / "tools" / "migrate.py").exists()
    makefile = (ROOT / "Makefile").read_text()
    assert not re.search(r"^(migrate|check):", makefile, re.M)
    leftovers = [p for p in (ROOT / "models").rglob("*") if p.name in ("model.lock.json", "migrations")]
    assert leftovers == []


@pytest.mark.parametrize("field,line", [
    ("immutable", "immutable: true\n"),
    ("mutable_columns", "mutable_columns: [kind]\n"),
    ("business_rules", "business_rules: []\n"),
    ("sample_rows", "sample_rows: []\n"),
])
def test_removed_table_field_rejected(workcopy, field, line):
    p = workcopy / "domains/audit/tables/event.yaml"
    p.write_text(p.read_text() + line)
    with pytest.raises(ValidationError, match=field):
        load(workcopy)


def test_removed_column_field_rejected(workcopy):
    p = workcopy / "domains/audit/tables/event.yaml"
    p.write_text(p.read_text().replace("{name: kind, type: text,", "{name: kind, type: text, enum: kinds,"))
    with pytest.raises(ValidationError, match="enum"):
        load(workcopy)


def test_uuid_column_rejected(workcopy):
    p = workcopy / "domains/audit/tables/event.yaml"
    p.write_text(p.read_text().replace("{name: kind, type: text,", "{name: kind, type: uuid,"))
    with pytest.raises(ValidationError, match="unsupported type 'uuid'"):
        load(workcopy)


def test_explorer_payload_has_no_removed_sections():
    data = explorer.payload(load(MINI))
    assert "enums" not in data
    for t in data["tables"]:
        assert not {"immutable", "mutable_columns", "rules", "samples"} & t.keys()
        assert all("enum" not in c for c in t["columns"])


def test_model_yaml_rejected(workcopy):
    (workcopy / "model.yaml").write_text("comments: false\n")
    with pytest.raises(ValueError, match="model.yaml"):
        load(workcopy)
