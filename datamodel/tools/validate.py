"""Cross-reference and standards checks. Exits non-zero on any error."""
from __future__ import annotations

import argparse
import re
import sys

from schema import OPTIONAL_SYSTEM
from model import DEFAULT_MODEL, Model, all_columns, con_name, load, model_dir

PG_IDENT_MAX = 63  # Postgres silently truncates longer identifiers, so the model stops matching the database.
RESERVED = {"user", "order", "group", "table", "column", "check", "default", "end", "select"}


def constraint_names(m: Model):
    """Yield (table, kind, generated_name) for everything the DDL generator names."""
    s = m.settings
    for t in m.tables.values():
        names = {c.name for c in all_columns(t, s.id_type)}
        yield t.table, "pk", con_name(s, "pk", t.table, [])
        if s.public_id_check and "public_id" in names:
            yield t.table, "check", f"ck_{t.table}_public_id"
        if s.updated_at_trigger and "updated_at" in names:
            yield t.table, "trigger", f"tg_{t.table}_updated_at"
        if t.immutable:
            yield t.table, "trigger", f"trg_{t.table}_immutable"
        for c in all_columns(t, s.id_type):
            if c.references:
                yield t.table, "fk", con_name(s, "fk", t.table, [c.name])
            if c.enum:
                yield t.table, "enum check", con_name(s, "enum", t.table, [c.name])
            if c.unique:
                yield t.table, "unique", con_name(s, "uq", t.table, [c.name])
        for cols in t.unique:
            yield t.table, "unique", con_name(s, "uq", t.table, cols)
        for ix in t.indexes:
            yield t.table, "index", f"ix_{t.table}_{'_'.join(ix.columns)}"
        for ck in t.checks:
            yield t.table, "check", ck.name


def validate(m: Model) -> list[str]:
    errs: list[str] = []
    e = errs.append

    # domains
    for d in m.domains.values():
        for dep in d.depends_on:
            if dep not in m.domains:
                e(f"domain {d.domain}: depends_on unknown domain {dep}")
            elif m.domains[dep].load_order >= d.load_order:
                e(f"domain {d.domain}: depends on {dep} which loads later or at the same order")
    orders = [d.load_order for d in m.domains.values()]
    if len(orders) != len(set(orders)):
        e("domains: load_order values must be unique")

    ids: dict[str, str] = {}
    prefixes: dict[str, str] = {}
    for t in m.tables.values():
        where = f"table {t.table}"
        if t.domain not in m.domains:
            e(f"{where}: unknown domain {t.domain}")
        if t.table in RESERVED:
            e(f"{where}: reserved word")
        for n in t.system_columns:
            if n not in OPTIONAL_SYSTEM:
                e(f"{where}: system_columns entry {n!r} must be one of {', '.join(OPTIONAL_SYSTEM)}")
        if len(set(t.system_columns)) != len(t.system_columns):
            e(f"{where}: system_columns has duplicates")
        if "public_id" not in t.system_columns:
            if t.public_id_prefix is not None:
                e(f"{where}: public_id_prefix set but public_id is not in system_columns")
        elif t.public_id_prefix is None:
            e(f"{where}: public_id_prefix is required when public_id is injected")
        else:
            if t.public_id_prefix in prefixes:
                e(f"{where}: public_id_prefix {t.public_id_prefix!r} already used by {prefixes[t.public_id_prefix]}")
            prefixes[t.public_id_prefix] = t.table
            if not re.fullmatch(r"[a-z]{2,6}", t.public_id_prefix):
                e(f"{where}: public_id_prefix must be 2 to 6 lowercase letters")

        if len(t.table.encode()) > PG_IDENT_MAX:
            e(f"{where}: table name is {len(t.table.encode())} bytes; Postgres truncates at {PG_IDENT_MAX}")
        for c in all_columns(t, m.settings.id_type):
            if len(c.name.encode()) > PG_IDENT_MAX:
                e(f"{where}.{c.name}: column name is {len(c.name.encode())} bytes; Postgres truncates at {PG_IDENT_MAX}")

        names = [c.name for c in all_columns(t, m.settings.id_type)]
        for n in {n for n in names if names.count(n) > 1}:
            e(f"{where}: duplicate or system-reserved column {n}")
        cols = {c.name: c for c in all_columns(t, m.settings.id_type)}

        for c in t.columns:
            if c.pii in ("direct", "phi") and c.default:
                e(f"{where}.{c.name}: a direct identifier or PHI column cannot have a default")
            if c.enum and c.enum not in m.enums:
                e(f"{where}.{c.name}: unknown enum {c.enum}")
            if c.enum and c.type != "text":
                e(f"{where}.{c.name}: enum columns must be text")
            if c.references:
                try:
                    rt, rc = c.references.split(".")
                except ValueError:
                    e(f"{where}.{c.name}: references must be 'table.column'")
                    continue
                if rt not in m.tables:
                    e(f"{where}.{c.name}: references unknown table {rt}")
                    continue
                if rc != "id":
                    e(f"{where}.{c.name}: foreign keys point at the internal id, not {rc}")
                fk_type = "bigint" if m.settings.id_type == "bigserial" else "uuid"
                if c.type != fk_type:
                    e(f"{where}.{c.name}: foreign key column must be {fk_type}")
                if c.on_delete == "set null" and not c.nullable:
                    e(f"{where}.{c.name}: on_delete 'set null' needs a nullable column")
                rd = m.tables[rt].domain
                deps = m.domains[t.domain].depends_on if t.domain in m.domains else []
                if rd != t.domain and rd not in deps:
                    e(f"{where}.{c.name}: references {rt} in domain {rd}, which {t.domain} does not depend on")
        for grp in t.unique:
            for n in grp:
                if n not in cols:
                    e(f"{where}: unique constraint names unknown column {n}")
        for ix in t.indexes:
            for n in ix.columns:
                if n not in cols:
                    e(f"{where}: index names unknown column {n}")
        for n in t.mutable_columns:
            if n not in cols:
                e(f"{where}: mutable_columns names unknown column {n}")
        if t.mutable_columns and not t.immutable:
            e(f"{where}: mutable_columns only makes sense on an immutable table")
        if t.immutable and any(c.on_delete != "restrict" for c in t.columns):
            e(f"{where}: an immutable table cannot rely on cascade or set null")

        for r in t.business_rules:
            if not r.id.startswith(f"BR-{t.table.upper()}-"):
                e(f"{where}: rule {r.id} must start with BR-{t.table.upper()}-")
        for q in t.open_questions:
            if not q.id.startswith(f"OQ-{t.table.upper()}-"):
                e(f"{where}: question {q.id} must start with OQ-{t.table.upper()}-")
            if q.status != "open" and not q.answer:
                e(f"{where}: question {q.id} is {q.status} but has no answer")
        for item in [*(r.id for r in t.business_rules), *(q.id for q in t.open_questions)]:
            if item in ids:
                e(f"{where}: id {item} already used in {ids[item]}")
            ids[item] = t.table
        if t.status == "published" and any(q.status == "open" and q.owner == "CMO" for q in t.open_questions):
            e(f"{where}: published with an open CMO question; clinical scope is the CMO's to settle")

        # sample rows: synthetic only, known columns only
        for i, row in enumerate(t.sample_rows):
            for k, v in row.items():
                if k not in cols:
                    e(f"{where}: sample row {i} has unknown column {k}")
                elif cols[k].enum and v not in m.enums.get(cols[k].enum, type("x", (), {"values": []})).values:
                    e(f"{where}: sample row {i} {k}={v!r} not in enum {cols[k].enum}")
                if isinstance(v, str) and re.search(r"\b\d{3}-\d{2}-\d{4}\b", v):
                    e(f"{where}: sample row {i} looks like an SSN. Synthetic data only")
                if k == "email" and isinstance(v, str) and not v.endswith((".invalid", "@example.com", "@example.org")):
                    e(f"{where}: sample row {i} email must use a reserved domain (.invalid, example.com)")

    # name length, collisions
    seen: dict[str, str] = {}
    for tbl, kind, name in constraint_names(m):
        if len(name.encode()) > PG_IDENT_MAX:
            e(f"table {tbl}: {kind} name {name!r} is {len(name.encode())} bytes; Postgres truncates at {PG_IDENT_MAX}")
        if name in seen and seen[name] != tbl:
            e(f"{name!r} used on both {seen[name]} and {tbl}")
        seen[name] = tbl
    for en in m.enums.values():
        if len(set(en.values)) != len(en.values):
            e(f"enum {en.enum}: duplicate values")
        if not all(re.fullmatch(r"[A-Z][A-Z0-9_]*", v) for v in en.values):
            e(f"enum {en.enum}: values must be UPPER_SNAKE")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    m = load(model_dir(ap.parse_args().model))
    errs = validate(m)
    for x in errs:
        print("ERROR", x, file=sys.stderr)
    print(f"{len(m.domains)} domains, {len(m.tables)} tables, {len(m.enums)} enums, {len(errs)} errors")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
