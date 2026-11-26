"""tools/drift.py: the diff and name-stripping need no database; the pg tests run it end to end."""
import copy
import shutil
import subprocess
import sys

import drift
import pytest
from conftest import ROOT
from test_postgres import URL, newdb, psql  # noqa: F401 (newdb is a fixture)

BASE = {"t": {"columns": {"id": ["bigint", True, None], "x": ["text", False, None]},
              "constraints": ["PRIMARY KEY (id)"], "indexes": [], "triggers": []}}


def test_identical_has_no_drift():
    assert drift.diff(BASE, copy.deepcopy(BASE)) == []


def test_extra_column_named():
    b = copy.deepcopy(BASE)
    b["t"]["columns"]["y"] = ["int", False, None]
    assert any("table t: column y only in b" in line for line in drift.diff(BASE, b))


def test_nullability_and_default_named():
    b = copy.deepcopy(BASE)
    b["t"]["columns"]["x"] = ["text", True, "'a'::text"]
    out = drift.diff(BASE, b)
    assert len(out) == 1 and "column x differs" in out[0]


def test_missing_table_and_constraint():
    b = copy.deepcopy(BASE)
    b["t"]["constraints"] = []
    b["u"] = BASE["t"]
    out = drift.diff(BASE, b)
    assert "table u: only in b" in out
    assert any("constraint only in a: PRIMARY KEY (id)" in line for line in out)


def test_names_stripped():
    ix, ix_sub = drift.STRIP["indexes"]
    assert ix.sub(ix_sub, "CREATE UNIQUE INDEX ix_a ON public.t USING btree (x)") == \
        "CREATE UNIQUE INDEX ON public.t USING btree (x)"
    tg, tg_sub = drift.STRIP["triggers"]
    assert tg.sub(tg_sub, "CREATE TRIGGER tg_a BEFORE UPDATE ON public.t FOR EACH ROW EXECUTE FUNCTION f()") == \
        "CREATE TRIGGER BEFORE UPDATE ON public.t FOR EACH ROW EXECUTE FUNCTION f()"


def _drift(a, b):
    return subprocess.run([sys.executable, str(ROOT / "tools" / "drift.py"), a, b], capture_output=True, text=True)


SCHEMA = """CREATE FUNCTION {f}() RETURNS trigger AS $$ BEGIN RETURN NEW; END $$ LANGUAGE plpgsql;
CREATE TABLE t (id bigserial PRIMARY KEY, x text NOT NULL, y text);
ALTER TABLE t ADD CONSTRAINT {c} CHECK (x <> '');
CREATE INDEX {i} ON t (y);
CREATE TRIGGER {g} BEFORE UPDATE ON t FOR EACH ROW EXECUTE FUNCTION {f}();
COMMENT ON TABLE t IS '{note}'"""


def _db(newdb, **names):
    url = newdb()
    for stmt in SCHEMA.format(**names).split(";\n"):
        psql(url, stmt)
    return url


@pytest.mark.pg
@pytest.mark.skipif(not URL or not shutil.which("psql"), reason="needs DATAMODEL_TEST_DATABASE_URL and psql")
def test_names_and_comments_are_not_drift(newdb):
    a = _db(newdb, f="f", c="ck_one", i="ix_one", g="tg_one", note="first")
    b = _db(newdb, f="f", c="ck_two", i="ix_two", g="tg_two", note="second")
    r = _drift(a, b)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "no drift" in r.stdout


@pytest.mark.pg
@pytest.mark.skipif(not URL or not shutil.which("psql"), reason="needs DATAMODEL_TEST_DATABASE_URL and psql")
def test_extra_column_is_drift(newdb):
    a = _db(newdb, f="f", c="c", i="i", g="g", note="n")
    b = _db(newdb, f="f", c="c", i="i", g="g", note="n")
    psql(b, "ALTER TABLE t ADD COLUMN extra_col int")
    r = _drift(a, b)
    assert r.returncode == 1 and "extra_col" in r.stdout


@pytest.mark.pg
@pytest.mark.skipif(not URL or not shutil.which("psql"), reason="needs DATAMODEL_TEST_DATABASE_URL and psql")
def test_nullability_change_is_drift(newdb):
    a = _db(newdb, f="f", c="c", i="i", g="g", note="n")
    b = _db(newdb, f="f", c="c", i="i", g="g", note="n")
    psql(b, "ALTER TABLE t ALTER COLUMN y SET NOT NULL")
    r = _drift(a, b)
    assert r.returncode == 1 and "column y" in r.stdout


def test_labels_name_the_sides():
    a, b = copy.deepcopy(BASE), copy.deepcopy(BASE)
    b["t"]["columns"]["y"] = ["int", False, None]
    a["t"]["constraints"].append("CHECK ((x <> ''::text))")
    a["t"]["columns"]["x"] = ["text", True, None]
    a["only_a"] = BASE["t"]
    out = drift.diff(a, b, "alembic", "yaml")
    assert "table only_a: only in alembic" in out
    assert any("column y only in yaml" in line for line in out)
    assert any("constraint only in alembic" in line for line in out)
    assert any("column x differs: alembic=" in line and " yaml=" in line for line in out)


def test_cli_labels(monkeypatch, capsys):
    a, b = copy.deepcopy(BASE), copy.deepcopy(BASE)
    b["t"]["columns"]["y"] = ["int", False, None]
    snaps = {"u1": a, "u2": b}
    monkeypatch.setattr(drift, "snapshot", snaps.get)
    monkeypatch.setattr(sys, "argv", ["drift.py", "u1", "u2", "--label-a", "alembic", "--label-b", "yaml"])
    assert drift.main() == 1
    assert "column y only in yaml" in capsys.readouterr().out
