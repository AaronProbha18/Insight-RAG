"""Compare the structure of two Postgres databases: columns, constraints, indexes, triggers.

Used by the plumbline-drift CI job: Alembic head in one database, the schema generated from
models/plumbline in the other. Object names, comments, column order and alembic_version are ignored.
Usage: drift.py <url-a> <url-b> [--label-a alembic --label-b yaml]
(exit 0 same, 1 drift, 2 psql failed). The labels name the sides in the output; default a and b.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

TABLES = ("c.relkind = 'r' AND n.nspname = 'public' AND c.relname <> 'alembic_version'"
          " AND c.relispartition = false")
FROM = "pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace"

QUERIES = {
    "columns": f"""SELECT c.relname AS tbl, a.attname AS name, format_type(a.atttypid, a.atttypmod) AS type,
        a.attnotnull AS notnull, pg_get_expr(d.adbin, d.adrelid) AS def
        FROM {FROM} JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
        LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum WHERE {TABLES}""",
    "constraints": f"""SELECT c.relname AS tbl, pg_get_constraintdef(k.oid) AS def
        FROM {FROM} JOIN pg_constraint k ON k.conrelid = c.oid AND k.contype <> 'n' WHERE {TABLES}""",
    "indexes": f"""SELECT c.relname AS tbl, pg_get_indexdef(i.indexrelid) AS def
        FROM {FROM} JOIN pg_index i ON i.indrelid = c.oid WHERE {TABLES}""",
    "triggers": f"""SELECT c.relname AS tbl, pg_get_triggerdef(t.oid) AS def
        FROM {FROM} JOIN pg_trigger t ON t.tgrelid = c.oid AND NOT t.tgisinternal WHERE {TABLES}""",
}
# Names differ between Alembic and the YAML model, so they come out of the definitions.
STRIP = {"indexes": (re.compile(r"^(CREATE (?:UNIQUE )?INDEX) \S+ ON "), r"\1 ON "),
         "triggers": (re.compile(r"^(CREATE (?:CONSTRAINT )?TRIGGER) \S+ "), r"\1 ")}


class PsqlError(RuntimeError):
    pass


def _rows(url: str, sql: str) -> list[dict]:
    r = subprocess.run(["psql", url, "-X", "-q", "-At", "-v", "ON_ERROR_STOP=1", "-c",
                        f"SELECT coalesce(json_agg(t), '[]') FROM ({sql}) t"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise PsqlError(r.stderr.strip())
    return json.loads(r.stdout)


def snapshot(url: str) -> dict:
    snap: dict[str, dict] = {}

    def table(name: str) -> dict:
        return snap.setdefault(name, {"columns": {}, "constraints": [], "indexes": [], "triggers": []})

    for row in _rows(url, QUERIES["columns"]):
        table(row["tbl"])["columns"][row["name"]] = [row["type"], row["notnull"], row["def"]]
    for kind in ("constraints", "indexes", "triggers"):
        for row in _rows(url, QUERIES[kind]):
            d = row["def"]
            if kind in STRIP:
                d = STRIP[kind][0].sub(STRIP[kind][1], d)
            table(row["tbl"])[kind].append(d)
    for t in snap.values():
        for kind in ("constraints", "indexes", "triggers"):
            t[kind].sort()
    return snap


def diff(a: dict, b: dict, label_a: str = "a", label_b: str = "b") -> list[str]:
    out: list[str] = []
    for name in sorted(a.keys() - b.keys()):
        out.append(f"table {name}: only in {label_a}")
    for name in sorted(b.keys() - a.keys()):
        out.append(f"table {name}: only in {label_b}")
    for name in sorted(a.keys() & b.keys()):
        ta, tb = a[name], b[name]
        for col in sorted(ta["columns"].keys() - tb["columns"].keys()):
            out.append(f"table {name}: column {col} only in {label_a} {ta['columns'][col]}")
        for col in sorted(tb["columns"].keys() - ta["columns"].keys()):
            out.append(f"table {name}: column {col} only in {label_b} {tb['columns'][col]}")
        for col in sorted(ta["columns"].keys() & tb["columns"].keys()):
            if ta["columns"][col] != tb["columns"][col]:
                out.append(f"table {name}: column {col} differs: "
                           f"{label_a}={ta['columns'][col]} {label_b}={tb['columns'][col]}")
        for kind in ("constraints", "indexes", "triggers"):
            for d in sorted(set(ta[kind]) - set(tb[kind])):
                out.append(f"table {name}: {kind[:-1]} only in {label_a}: {d}")
            for d in sorted(set(tb[kind]) - set(ta[kind])):
                out.append(f"table {name}: {kind[:-1]} only in {label_b}: {d}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url_a")
    ap.add_argument("url_b")
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    args = ap.parse_args()
    try:
        lines = diff(snapshot(args.url_a), snapshot(args.url_b), args.label_a, args.label_b)
    except PsqlError as e:
        print(f"psql failed: {e}", file=sys.stderr)
        return 2
    print("\n".join(lines) if lines else "no drift")
    return 1 if lines else 0


if __name__ == "__main__":
    sys.exit(main())
