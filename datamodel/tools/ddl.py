"""Generate Postgres DDL from the model.

build_objects() returns a plain dict describing every database object; render_full() renders it
as the DDL the plumbline-drift job loads into Postgres.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from model import DEFAULT_MODEL, Model, all_columns, column_type, con_name, load, model_dir

HEADER = "-- GENERATED from domains/**/*.yaml. Do not edit. Change the YAML and run `make ddl`.\n"


def build_objects(m: Model) -> dict:
    tables: dict[str, dict] = {}
    for t in m.ordered_tables():
        cols = all_columns(t)
        # id, public_id first; created_at/updated_at last
        head = [c for c in cols if c.name in ("id", "public_id")]
        tail = [c for c in cols if c.name in ("created_at", "updated_at")]
        body = [c for c in cols if c not in head and c not in tail]
        ordered = head + body + tail

        columns: dict[str, dict] = {}
        for c in ordered:
            d = f"{c.name} {column_type(c)}"
            if not c.nullable:
                d += " NOT NULL"
            if c.default is not None:
                d += f" DEFAULT {c.default}"
            columns[c.name] = {"def": d}

        cons: dict[str, str] = {con_name("pk", t.table, []): "PRIMARY KEY (id)"}
        for c in cols:
            if c.unique:
                cons[con_name("uq", t.table, [c.name])] = f"UNIQUE ({c.name})"
            if c.references:
                rt, rc = c.references.split(".")
                cons[con_name("fk", t.table, [c.name])] = (
                    f"FOREIGN KEY ({c.name}) REFERENCES {rt} ({rc}) ON DELETE {c.on_delete.upper()}")
        for grp in t.unique:
            cons[con_name("uq", t.table, grp)] = f"UNIQUE ({', '.join(grp)})"
        for ck in t.checks:
            cons[ck.name] = f"CHECK ({ck.expr})"

        idx: dict[str, str] = {}
        for ix in t.indexes:
            name = f"ix_{t.table}_{'_'.join(ix.columns)}"
            sql = f"CREATE {'UNIQUE ' if ix.unique else ''}INDEX {name} ON {t.table} ({', '.join(ix.columns)})"
            if ix.where:
                sql += f" WHERE {ix.where}"
            idx[name] = sql + ";"

        tables[t.table] = {"domain": t.domain, "columns": columns, "constraints": cons, "indexes": idx}
    return {"tables": tables}


def create_table_sql(name: str, t: dict) -> tuple[list[str], list[str]]:
    """CREATE TABLE plus everything hanging off it. FKs come last so order never matters."""
    cols = ",\n  ".join(c["def"] for c in t["columns"].values())
    out = [f"CREATE TABLE {name} (\n  {cols}\n);"]
    fks = []
    for n, sql in t["constraints"].items():
        stmt = f"ALTER TABLE {name} ADD CONSTRAINT {n} {sql};"
        (fks if sql.startswith("FOREIGN KEY") else out).append(stmt)
    out += list(t["indexes"].values())
    return out, fks


def render_full(objs: dict) -> str:
    parts = [HEADER]
    all_fks: list[str] = []
    for name, t in objs["tables"].items():
        body, fks = create_table_sql(name, t)
        parts.append(f"-- {t['domain']}.{name}\n" + "\n".join(body) + "\n")
        all_fks += fks
    parts.append("-- foreign keys\n" + "\n".join(all_fks) + "\n")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    md = model_dir(ap.parse_args().model)
    m = load(md)
    out = md / "build" / "schema.sql"
    out.parent.mkdir(exist_ok=True)
    objs = build_objects(m)
    out.write_text(render_full(objs))
    print(f"wrote {out} ({len(objs['tables'])} tables)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
