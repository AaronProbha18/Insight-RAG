"""Load domains/**/*.yaml and enums/*.yaml into one in-memory Model."""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field

import yaml

from schema import Column, Domain, EnumDef, Settings, Table

ROOT = pathlib.Path(__file__).resolve().parent.parent  # the toolkit; models live under models/<name>/
DEFAULT_MODEL = "models/plumbline"


def model_dir(name: str = DEFAULT_MODEL) -> pathlib.Path:
    return ROOT / name  # an absolute name wins over ROOT


def system_columns(t: Table, id_type: str = "uuid") -> list[Column]:
    """Columns every table gets, so no YAML can forget them or contradict the standard."""
    pk = Column(name="id", type="uuid", default="gen_random_uuid()",
                description="Internal primary key. Never leaves the backend.")
    if id_type == "bigserial":  # a key-only pseudo-type, deliberately not a YAML column type
        pk = pk.model_copy(update={"type": "bigserial", "default": None})
    cols = [pk]
    if "public_id" in t.system_columns:
        cols.append(Column(name="public_id", type="text", unique=True,
                           description=f"Opaque external identifier, prefixed '{t.public_id_prefix}_'."))
    if "created_at" in t.system_columns:
        cols.append(Column(name="created_at", type="timestamptz", default="now()",
                           description="When the row was written."))
    if "updated_at" in t.system_columns and (not t.immutable or t.mutable_columns):
        cols.append(Column(name="updated_at", type="timestamptz", default="now()",
                           description="When the row last changed."))
    return cols


def column_type(c: Column) -> str:
    """The SQL type, with {embedding_dim} filled from PLUMBLINE_EMBEDDING_DIM (same rule as backend migration 0001)."""
    if "{embedding_dim}" not in c.type:
        return c.type
    value = os.environ.get("PLUMBLINE_EMBEDDING_DIM", "")
    if not value.isdigit() or int(value) == 0:
        raise ValueError(f"{c.name}: PLUMBLINE_EMBEDDING_DIM must be set to a positive integer; got {value!r}")
    return c.type.replace("{embedding_dim}", value)


def all_columns(t: Table, id_type: str = "uuid") -> list[Column]:
    return system_columns(t, id_type) + t.columns


def con_name(s: Settings, kind: str, table: str, cols: list[str]) -> str:
    """Constraint name per model.yaml constraint_names: ours (pk_/fk_/uq_/ck_) or Postgres's defaults."""
    c = "_".join(cols)
    if s.constraint_names == "postgres":
        return {"pk": f"{table}_pkey", "fk": f"{table}_{c}_fkey", "uq": f"{table}_{c}_key",
                "enum": f"{table}_{c}_check"}[kind]
    return {"pk": f"pk_{table}", "fk": f"fk_{table}_{c}", "uq": f"uq_{table}_{c}",
            "enum": f"ck_{table}_{c}_enum"}[kind]


@dataclass
class Model:
    domains: dict[str, Domain] = field(default_factory=dict)
    tables: dict[str, Table] = field(default_factory=dict)
    enums: dict[str, EnumDef] = field(default_factory=dict)
    settings: Settings = field(default_factory=Settings)

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


def load(root: pathlib.Path | None = None) -> Model:
    root = root or model_dir()
    m = Model()
    if (root / "model.yaml").exists():
        m.settings = Settings(**(_yaml(root / "model.yaml") or {}))
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
