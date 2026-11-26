"""Plumbline's conventions, per-table system_columns and the extra column types, on a temp copy of the mini fixture."""
import pytest

import ddl
import validate
from conftest import MINI
from model import load


def test_plumbline_conventions_generated():
    sql = ddl.render_full(ddl.build_objects(load(MINI)))
    assert "id bigserial NOT NULL" in sql and "account_id bigint NOT NULL" in sql
    assert "_public_id CHECK" not in sql
    assert "BEFORE UPDATE" not in sql and "set_updated_at()" not in sql
    assert "COMMENT ON" not in sql
    assert "ADD CONSTRAINT account_pkey PRIMARY KEY" in sql
    assert "ADD CONSTRAINT account_public_id_key UNIQUE" in sql
    assert "ADD CONSTRAINT note_account_id_fkey FOREIGN KEY" in sql
    for old in ("ADD CONSTRAINT pk_", "ADD CONSTRAINT fk_", "ADD CONSTRAINT uq_"):
        assert old not in sql


def test_non_bigint_foreign_key_rejected(workcopy):
    p = workcopy / "domains/core/tables/note.yaml"
    p.write_text(p.read_text().replace("type: bigint", "type: int"))
    errs = validate.validate(load(workcopy))
    assert any("note.account_id" in e and "must be bigint" in e for e in errs)


def _table(model_dir, extra: str, drop_prefix=False):
    p = model_dir / "domains/core/tables/account.yaml"
    text = p.read_text()
    if drop_prefix:
        text = "\n".join(line for line in text.splitlines() if not line.startswith("public_id_prefix:")) + "\n"
    p.write_text(text + extra)


def test_system_columns_subset(workcopy):
    _table(workcopy, "system_columns: [created_at]\n", drop_prefix=True)
    m = load(workcopy)
    assert validate.validate(m) == []
    acc = ddl.build_objects(m)["tables"]["account"]
    assert [c for c in acc["columns"] if c in ("id", "public_id", "created_at", "updated_at")] == ["id", "created_at"]
    assert not any("public_id" in n for n in acc["constraints"])
    assert not acc["triggers"]


@pytest.mark.parametrize("entry", ["id", "bogus"])
def test_system_columns_bad_entry(workcopy, entry):
    _table(workcopy, f"system_columns: [{entry}]\n")
    errs = validate.validate(load(workcopy))
    assert any("table account: system_columns" in e and entry in e for e in errs)


def test_prefix_only_with_public_id(workcopy):
    _table(workcopy, "system_columns: [created_at]\n")
    assert any("public_id_prefix set" in e for e in validate.validate(load(workcopy)))


def _typed(model_dir):
    p = model_dir / "domains/core/tables/account.yaml"
    p.write_text(p.read_text().replace("columns:\n", "columns:\n"
        "  - {name: embedding, type: 'vector({embedding_dim})', description: Synthetic test vector.}\n"
        "  - {name: digest, type: char(64), description: Synthetic hash.}\n"
        "  - {name: score, type: double precision, description: Synthetic score.}\n", 1))
    return load(model_dir)


def test_embedding_dim_filled(workcopy, monkeypatch):
    monkeypatch.setenv("PLUMBLINE_EMBEDDING_DIM", "8")
    cols = ddl.build_objects(_typed(workcopy))["tables"]["account"]["columns"]
    assert cols["embedding"]["def"].startswith("embedding vector(8) ")
    assert cols["digest"]["def"].startswith("digest char(64) ")
    assert cols["score"]["def"].startswith("score double precision ")


@pytest.mark.parametrize("value", [None, "0"])
def test_embedding_dim_missing(workcopy, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("PLUMBLINE_EMBEDDING_DIM", raising=False)
    else:
        monkeypatch.setenv("PLUMBLINE_EMBEDDING_DIM", value)
    m = _typed(workcopy)
    with pytest.raises(ValueError, match="PLUMBLINE_EMBEDDING_DIM"):
        ddl.build_objects(m)


def test_validate_needs_no_embedding_dim(workcopy, monkeypatch):
    monkeypatch.delenv("PLUMBLINE_EMBEDDING_DIM", raising=False)
    assert validate.validate(_typed(workcopy)) == []
