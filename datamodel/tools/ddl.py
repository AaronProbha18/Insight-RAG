"""Generate Postgres DDL from the model.

build_objects() returns a plain dict describing every database object. Full DDL renders it
whole; migrate.py diffs two of them. One source for both keeps migrations honest.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from model import DEFAULT_MODEL, Model, all_columns, con_name, load, model_dir

HEADER = "-- GENERATED from domains/**/*.yaml. Do not edit. Change the YAML and run `make ddl`.\n"

SET_UPDATED_AT = """CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;"""


def q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def build_objects(m: Model) -> dict:
    s = m.settings
    tables: dict[str, dict] = {}
    for t in m.ordered_tables():
        cols = all_columns(t, s.id_type)
        # id, public_id first; created_at/updated_at last
        head = [c for c in cols if c.name in ("id", "public_id")]
        tail = [c for c in cols if c.name in ("created_at", "updated_at")]
        body = [c for c in cols if c not in head and c not in tail]
        ordered = head + body + tail

        columns: dict[str, dict] = {}
        for c in ordered:
            d = f"{c.name} {c.type}"
            if not c.nullable:
                d += " NOT NULL"
            if c.default is not None:
                d += f" DEFAULT {c.default}"
            comment = c.description + (f" [pii: {c.pii}]" if c.pii != "none" else "")
            columns[c.name] = {"def": d, "comment": comment if s.comments else None}

        cons: dict[str, str] = {con_name(s, "pk", t.table, []): "PRIMARY KEY (id)"}
        if s.public_id_check and "public_id" in columns:
            cons[f"ck_{t.table}_public_id"] = f"CHECK (public_id ~ '^{t.public_id_prefix}_[0-9A-Za-z]+$')"
        for c in cols:
            if c.unique:
                cons[con_name(s, "uq", t.table, [c.name])] = f"UNIQUE ({c.name})"
            if c.enum:
                vals = ", ".join(q(v) for v in m.enums[c.enum].values)
                cons[con_name(s, "enum", t.table, [c.name])] = f"CHECK ({c.name} IN ({vals}))"
            if c.references:
                rt, rc = c.references.split(".")
                cons[con_name(s, "fk", t.table, [c.name])] = (
                    f"FOREIGN KEY ({c.name}) REFERENCES {rt} ({rc}) ON DELETE {c.on_delete.upper()}")
        for grp in t.unique:
            cons[con_name(s, "uq", t.table, grp)] = f"UNIQUE ({', '.join(grp)})"
        for ck in t.checks:
            cons[ck.name] = f"CHECK ({ck.expr})"

        idx: dict[str, str] = {}
        for ix in t.indexes:
            name = f"ix_{t.table}_{'_'.join(ix.columns)}"
            sql = f"CREATE {'UNIQUE ' if ix.unique else ''}INDEX {name} ON {t.table} ({', '.join(ix.columns)})"
            if ix.where:
                sql += f" WHERE {ix.where}"
            idx[name] = sql + ";"

        trig: dict[str, str] = {}
        if "updated_at" in columns and s.updated_at_trigger:
            trig[f"tg_{t.table}_updated_at"] = (
                f"CREATE TRIGGER tg_{t.table}_updated_at BEFORE UPDATE ON {t.table} "
                f"FOR EACH ROW EXECUTE FUNCTION set_updated_at();")
        if t.immutable:
            allowed = list(t.mutable_columns) + (["updated_at"] if "updated_at" in columns else [])
            arr = "ARRAY[" + ", ".join(q(a) for a in allowed) + "]::text[]" if allowed else "ARRAY[]::text[]"
            fn = f"trg_{t.table}_immutable"
            trig[fn] = (
                f"CREATE OR REPLACE FUNCTION {fn}() RETURNS trigger AS $$\n"
                f"BEGIN\n"
                f"  IF TG_OP = 'DELETE' THEN\n"
                f"    RAISE EXCEPTION '{t.table} is immutable: rows cannot be deleted';\n"
                f"  END IF;\n"
                f"  IF (to_jsonb(NEW) - {arr}) IS DISTINCT FROM (to_jsonb(OLD) - {arr}) THEN\n"
                f"    RAISE EXCEPTION '{t.table} is immutable: only {', '.join(allowed) or 'no columns'} may change';\n"
                f"  END IF;\n"
                f"  RETURN NEW;\n"
                f"END;\n"
                f"$$ LANGUAGE plpgsql;\n"
                f"CREATE TRIGGER {fn} BEFORE UPDATE OR DELETE ON {t.table} "
                f"FOR EACH ROW EXECUTE FUNCTION {fn}();")

        tables[t.table] = {"domain": t.domain, "comment": t.description if s.comments else None,
                           "columns": columns,
                           "constraints": cons, "indexes": idx, "triggers": trig}
    return {"tables": tables}


def create_table_sql(name: str, t: dict) -> tuple[list[str], list[str]]:
    """CREATE TABLE plus everything hanging off it. FKs come last so order never matters."""
    cols = ",\n  ".join(c["def"] for c in t["columns"].values())
    out = [f"CREATE TABLE {name} (\n  {cols}\n);"]
    if t["comment"] is not None:
        out.append(f"COMMENT ON TABLE {name} IS {q(t['comment'])};")
    for cn, c in t["columns"].items():
        if c["comment"] is not None:
            out.append(f"COMMENT ON COLUMN {name}.{cn} IS {q(c['comment'])};")
    fks = []
    for n, sql in t["constraints"].items():
        stmt = f"ALTER TABLE {name} ADD CONSTRAINT {n} {sql};"
        (fks if sql.startswith("FOREIGN KEY") else out).append(stmt)
    out += list(t["indexes"].values())
    out += list(t["triggers"].values())
    return out, fks


def render_full(objs: dict) -> str:
    parts = [HEADER, "CREATE EXTENSION IF NOT EXISTS pgcrypto;\n"]
    if any(n.startswith("tg_") for t in objs["tables"].values() for n in t["triggers"]):
        parts.append(SET_UPDATED_AT + "\n")
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
