"""DDL generation and the tools' --model default, without Postgres."""
import os
import shutil
import subprocess
import sys

import ddl
from conftest import MINI, ROOT, run_tool
from model import load

SYSTEM = ("id", "public_id", "created_at", "updated_at")


def test_system_columns_injected():
    m = load(MINI)
    assert not any(c.name in SYSTEM for t in m.tables.values() for c in t.columns), "system column written in YAML"
    t = ddl.build_objects(m)["tables"]
    acc = t["account"]
    cols = list(acc["columns"])
    assert cols[:2] == ["id", "public_id"] and cols[-2:] == ["created_at", "updated_at"]
    assert acc["columns"]["id"]["def"] == "id bigserial NOT NULL"
    assert acc["constraints"]["account_pkey"] == "PRIMARY KEY (id)"
    assert acc["constraints"]["account_public_id_key"] == "UNIQUE (public_id)"
    assert list(t["note"]["columns"])[-1] == "created_at" and "updated_at" not in t["note"]["columns"]
    assert list(t["event"]["columns"]) == ["id", "kind", "created_at"]


def test_ddl_model_flag_writes_into_model_build(workcopy, tmp_path):
    other = tmp_path / "other"
    shutil.copytree(workcopy, other)
    r = run_tool(workcopy, "ddl.py", "--model", str(other))
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (workcopy / "build").exists()
    assert (other / "build/schema.sql").read_text() == ddl.render_full(ddl.build_objects(load(MINI)))


def test_default_model_is_plumbline(tmp_path):
    work = tmp_path / "dm"
    shutil.copytree(ROOT / "tools", work / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "models" / "plumbline", work / "models" / "plumbline", ignore=shutil.ignore_patterns("build"))
    env = {**os.environ, "PYTHONPATH": str(work / "tools"), "PLUMBLINE_EMBEDDING_DIM": "8"}
    for script in ("validate.py", "ddl.py"):
        r = subprocess.run([sys.executable, str(work / "tools" / script)], capture_output=True, text=True, env=env, cwd=work)
        assert r.returncode == 0, r.stdout + r.stderr
    assert "CREATE TABLE feedback" in (work / "models/plumbline/build/schema.sql").read_text()
