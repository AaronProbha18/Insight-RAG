"""Write the next migration by diffing the current model against model.lock.json.

model.lock.json is the model as of the last migration written. It is committed with the migration,
so a migration and the model state it produced always travel together.

Safe changes (new table, new nullable column, new constraint or index) are emitted directly.
Anything that can lose data or fail on existing rows is DESTRUCTIVE: the run is refused unless
--allow-destructive is passed, and the migration still needs a human to read it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

from ddl import SET_UPDATED_AT, build_objects, create_table_sql, render_full
from model import DEFAULT_MODEL, load, model_dir


def diff(old: dict, new: dict) -> tuple[list[str], list[str]]:
    """Return (statements, destructive_reasons)."""
    stmts: list[str] = []
    bad: list[str] = []
    ot, nt = old.get("tables", {}), new["tables"]
    fks: list[str] = []

    for name in nt.keys() - ot.keys():
        body, f = create_table_sql(name, nt[name])
        stmts += [f"-- new table {name}"] + body
        fks += f
    for name in ot.keys() - nt.keys():
        bad.append(f"drop table {name}")
        stmts.append(f"DROP TABLE {name};  -- DESTRUCTIVE")

    for name in nt.keys() & ot.keys():
        o, n = ot[name], nt[name]
        for cn in n["columns"].keys() - o["columns"].keys():
            c = n["columns"][cn]
            if "NOT NULL" in c["def"] and "DEFAULT" not in c["def"]:
                bad.append(f"{name}.{cn}: NOT NULL column with no default fails on a table that has rows")
            stmts.append(f"ALTER TABLE {name} ADD COLUMN {c['def']};")
            stmts.append(f"COMMENT ON COLUMN {name}.{cn} IS '{c['comment'].replace(chr(39), chr(39)*2)}';")
        for cn in o["columns"].keys() - n["columns"].keys():
            bad.append(f"drop column {name}.{cn}")
            stmts.append(f"ALTER TABLE {name} DROP COLUMN {cn};  -- DESTRUCTIVE")
        for cn in n["columns"].keys() & o["columns"].keys():
            od, nd = o["columns"][cn]["def"], n["columns"][cn]["def"]
            if od != nd:
                ot_, nt_ = _parts(od), _parts(nd)
                if ot_["type"] != nt_["type"]:
                    bad.append(f"{name}.{cn}: type {ot_['type']} -> {nt_['type']}")
                    stmts.append(f"ALTER TABLE {name} ALTER COLUMN {cn} TYPE {nt_['type']};  -- DESTRUCTIVE")
                if ot_["notnull"] != nt_["notnull"]:
                    if nt_["notnull"]:
                        bad.append(f"{name}.{cn}: SET NOT NULL fails if any row is null")
                    stmts.append(f"ALTER TABLE {name} ALTER COLUMN {cn} "
                                 f"{'SET' if nt_['notnull'] else 'DROP'} NOT NULL;")
                if ot_["default"] != nt_["default"]:
                    stmts.append(f"ALTER TABLE {name} ALTER COLUMN {cn} "
                                 + (f"SET DEFAULT {nt_['default']};" if nt_["default"] else "DROP DEFAULT;"))
            if o["columns"][cn]["comment"] != n["columns"][cn]["comment"]:
                stmts.append(f"COMMENT ON COLUMN {name}.{cn} IS "
                             f"'{n['columns'][cn]['comment'].replace(chr(39), chr(39)*2)}';")
        # constraints: changed = drop + add (enum value changes land here)
        for cn in o["constraints"].keys() - n["constraints"].keys():
            stmts.append(f"ALTER TABLE {name} DROP CONSTRAINT {cn};")
        for cn, sql in n["constraints"].items():
            if cn in o["constraints"] and o["constraints"][cn] == sql:
                continue
            if cn in o["constraints"]:
                stmts.append(f"ALTER TABLE {name} DROP CONSTRAINT {cn};")
            add = f"ALTER TABLE {name} ADD CONSTRAINT {cn} {sql};"
            (fks if sql.startswith("FOREIGN KEY") else stmts).append(add)
        for ix in o["indexes"].keys() - n["indexes"].keys():
            stmts.append(f"DROP INDEX {ix};")
        for ix, sql in n["indexes"].items():
            if o["indexes"].get(ix) != sql:
                if ix in o["indexes"]:
                    stmts.append(f"DROP INDEX {ix};")
                stmts.append(sql)
        for tg in o["triggers"].keys() - n["triggers"].keys():
            if tg.startswith("trg_"):
                stmts.append(f"DROP TRIGGER IF EXISTS {tg} ON {name};\nDROP FUNCTION IF EXISTS {tg}();")
            else:
                stmts.append(f"DROP TRIGGER IF EXISTS {tg} ON {name};")
        for tg, sql in n["triggers"].items():
            if o["triggers"].get(tg) != sql:
                if tg in o["triggers"]:
                    stmts.append(f"DROP TRIGGER IF EXISTS {tg} ON {name};")
                stmts.append(sql)
    return stmts + (["-- foreign keys"] + fks if fks else []), bad


def _parts(coldef: str) -> dict:
    m = re.match(r"^(\S+) (\S+(?: ?\(\d+(?:,\d+)?\))?)(.*)$", coldef)
    rest = m.group(3)
    d = re.search(r"DEFAULT (.+)$", rest)
    return {"type": m.group(2), "notnull": "NOT NULL" in rest, "default": d.group(1) if d else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="slug for the migration file")
    ap.add_argument("--allow-destructive", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    md = model_dir(a.model)
    LOCK, MIG = md / "model.lock.json", md / "migrations"

    new = build_objects(load(md))
    old = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    stmts, bad = diff(old, new)
    if not stmts:
        print("no changes")
        return 0
    if bad and not a.allow_destructive:
        print("REFUSED: destructive or risky changes. Review, then re-run with --allow-destructive:", file=sys.stderr)
        for b in bad:
            print("  -", b, file=sys.stderr)
        return 2

    MIG.mkdir(exist_ok=True)
    n = len(list(MIG.glob("[0-9][0-9][0-9][0-9]_*.sql"))) + 1
    slug = a.name or ("baseline" if not old else "change")
    path = MIG / f"{n:04d}_{slug}.sql"
    head = f"-- Migration {n:04d}. Generated by tools/migrate.py; review before applying.\n"
    if bad:
        head += "-- RISKY, approved with --allow-destructive:\n" + "".join(f"--   {b}\n" for b in bad)
    if not old:
        body = render_full(new)
    else:
        body = "\n".join(stmts) + "\n"
        if any(s.startswith("CREATE TRIGGER tg_") for s in stmts) or any("set_updated_at()" in s for s in stmts):
            body = SET_UPDATED_AT + "\n\n" + body
    if a.dry_run:
        print(head + body)
        return 0
    path.write_text(head + "BEGIN;\n" + body + "COMMIT;\n")
    LOCK.write_text(json.dumps(new, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path}; updated {LOCK.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
