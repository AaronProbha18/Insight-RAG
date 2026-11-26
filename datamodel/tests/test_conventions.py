"""model.yaml conventions, each on a temp copy of the sample model."""
import pytest
from pydantic import ValidationError

import ddl
import validate
from conftest import run_tool
from model import load


def _ddl(model_dir, settings: str) -> str:
    (model_dir / "model.yaml").write_text(settings)
    return ddl.render_full(ddl.build_objects(load(model_dir)))


def test_no_model_yaml_keeps_lock(workcopy):
    assert not (workcopy / "model.yaml").exists()
    assert run_tool(workcopy, "migrate.py").stdout.startswith("no changes")


def test_bigserial_ids_and_bigint_fks(workcopy):
    (workcopy / "model.yaml").write_text("id_type: bigserial\n")
    errs = validate.validate(load(workcopy))
    assert errs and all("must be bigint" in e for e in errs)

    for p in (workcopy / "domains").glob("*/tables/*.yaml"):
        p.write_text(p.read_text().replace("type: uuid", "type: bigint"))
    m = load(workcopy)
    assert validate.validate(m) == []
    assert ddl.build_objects(m)["tables"]["organization"]["columns"]["id"]["def"] == "id bigserial NOT NULL"


def test_public_id_check_off(workcopy):
    assert "_public_id CHECK" not in _ddl(workcopy, "public_id_check: false\n")


def test_updated_at_trigger_off(workcopy):
    sql = _ddl(workcopy, "updated_at_trigger: false\n")
    assert "_updated_at BEFORE UPDATE" not in sql and "set_updated_at()" not in sql
    assert "trg_person_consent_grant_immutable" in sql


def test_comments_off(workcopy):
    assert "COMMENT ON" not in _ddl(workcopy, "comments: false\n")


def test_postgres_constraint_names(workcopy):
    sql = _ddl(workcopy, "constraint_names: postgres\n")
    assert "CONSTRAINT organization_pkey PRIMARY KEY" in sql
    assert "CONSTRAINT organization_public_id_key UNIQUE" in sql
    assert "_fkey FOREIGN KEY" in sql
    for old in ("CONSTRAINT pk_", "CONSTRAINT fk_", "CONSTRAINT uq_"):
        assert old not in sql
    assert validate.validate(load(workcopy)) == []


def test_unknown_key_rejected(workcopy):
    (workcopy / "model.yaml").write_text("id_typo: bigserial\n")
    with pytest.raises(ValidationError, match="id_typo"):
        load(workcopy)
