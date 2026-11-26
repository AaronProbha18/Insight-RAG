"""Load domains/**/*.yaml and enums/*.yaml into one in-memory Model."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import yaml

from schema import Column, Domain, EnumDef, Table

ROOT = pathlib.Path(__file__).resolve().parent.parent


def system_columns(t: Table) -> list[Column]:
    """Columns every table gets, so no YAML can forget them or contradict the standard."""
    cols = [
        Column(name="id", type="uuid", default="gen_random_uuid()",
               description="Internal primary key. Never leaves the backend."),
        Column(name="public_id", type="text", unique=True,
               description=f"Opaque external identifier, prefixed '{t.public_id_prefix}_'."),
        Column(name="created_at", type="timestamptz", default="now()",
               description="When the row was written."),
    ]
    if not t.immutable or t.mutable_columns:
        cols.append(Column(name="updated_at", type="timestamptz", default="now()",
                           description="When the row last changed."))
    return cols


def all_columns(t: Table) -> list[Column]:
    return system_columns(t) + t.columns


@dataclass
class Model:
    domains: dict[str, Domain] = field(default_factory=dict)
    tables: dict[str, Table] = field(default_factory=dict)
    enums: dict[str, EnumDef] = field(default_factory=dict)

    def ordered_tables(self) -> list[Table]:
        """Domain load order first, then FK dependency order inside a domain, then name."""
        out: list[Table] = []
        seen: set[str] = set()

        def visit(t: Table, stack: tuple[str, ...] = ()) -> None:
            if t.table in seen or t.table in stack:
                return
            for c in t.columns:
                if c.references:
                    ref = c.references.split(".")[0]
                    if ref in self.tables and ref != t.table:
                        visit(self.tables[ref], stack + (t.table,))
            seen.add(t.table)
            out.append(t)

        for d in sorted(self.domains.values(), key=lambda d: d.load_order):
            for t in sorted((t for t in self.tables.values() if t.domain == d.domain),
                            key=lambda t: t.table):
                visit(t)
        return out


def _yaml(path: pathlib.Path):
    with path.open() as f:
        return yaml.safe_load(f)


def load(root: pathlib.Path = ROOT) -> Model:
    m = Model()
    for p in sorted((root / "domains").glob("*/domain.yaml")):
        d = Domain(**_yaml(p))
        m.domains[d.domain] = d
    for p in sorted((root / "domains").glob("*/tables/*.yaml")):
        t = Table(**_yaml(p))
        if p.stem != t.table:
            raise ValueError(f"{p}: file name must equal table name ({t.table})")
        if t.table in m.tables:
            raise ValueError(f"duplicate table {t.table}")
        m.tables[t.table] = t
    for p in sorted((root / "enums").glob("*.yaml")):
        e = EnumDef(**_yaml(p))
        m.enums[e.enum] = e
    return m
