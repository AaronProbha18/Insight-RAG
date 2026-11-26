"""Pydantic schema for the YAML model. The YAML is the model; everything else is generated."""
from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

NAME = re.compile(r"^[a-z][a-z0-9_]*$")
# Logical types allowed in YAML. Anything else is rejected rather than passed through to SQL.
TYPE = re.compile(
    r"^(text|int|bigint|bool|date|time|timestamptz|jsonb|uuid|inet|"
    r"numeric\(\d{1,2},\d{1,2}\)|varchar\(\d{1,4}\))$"
)
OPTIONAL_SYSTEM = ("public_id", "created_at", "updated_at")  # `id` is always injected
PII = Literal["none", "indirect", "direct", "phi"]
Layer = Literal["configuration", "reference", "runtime"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Column(Strict):
    name: str
    type: str
    nullable: bool = False
    pii: PII = "none"
    description: str
    default: Optional[str] = None  # raw SQL expression, e.g. "true", "now()"
    enum: Optional[str] = None  # name of an enum in enums/
    references: Optional[str] = None  # "table.column"
    on_delete: Literal["restrict", "cascade", "set null"] = "restrict"
    unique: bool = False

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not NAME.match(v):
            raise ValueError(f"bad column name {v!r}")
        return v

    @field_validator("type")
    @classmethod
    def _type(cls, v: str) -> str:
        if not TYPE.match(v):
            raise ValueError(f"unsupported type {v!r}")
        return v


class Check(Strict):
    name: str
    expr: str
    description: str = ""


class Index(Strict):
    columns: list[str]
    unique: bool = False
    where: Optional[str] = None


class Rule(Strict):
    id: str
    text: str
    enforced_by: Literal["db", "service", "process"] = "db"


class Question(Strict):
    id: str
    owner: Literal["CTO", "CMO", "Product", "CEO", "Business Tech Review"]
    status: Literal["open", "answered", "closed"] = "open"
    text: str
    answer: Optional[str] = None


class Table(Strict):
    table: str
    domain: str
    status: Literal["draft", "published"] = "draft"
    description: str
    # Two-identifier model: internal `id` (UUID) never leaves the backend,
    # `public_id` (prefixed, opaque) is the only identifier exposed externally.
    public_id_prefix: Optional[str] = None  # required when public_id is injected
    system_columns: list[str] = Field(default_factory=lambda: list(OPTIONAL_SYSTEM))
    immutable: bool = False
    mutable_columns: list[str] = Field(default_factory=list)  # allow-list on immutable tables
    columns: list[Column]
    checks: list[Check] = Field(default_factory=list)
    unique: list[list[str]] = Field(default_factory=list)
    indexes: list[Index] = Field(default_factory=list)
    business_rules: list[Rule] = Field(default_factory=list)
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)
    open_questions: list[Question] = Field(default_factory=list)

    @field_validator("table")
    @classmethod
    def _t(cls, v: str) -> str:
        if not NAME.match(v):
            raise ValueError(f"bad table name {v!r}")
        return v


class Domain(Strict):
    domain: str
    title: str
    layer: Layer
    load_order: int
    depends_on: list[str] = Field(default_factory=list)
    description: str


class Settings(Strict):
    """model.yaml: per-model conventions. Every default is the sample model's behaviour."""
    id_type: Literal["uuid", "bigserial"] = "uuid"
    public_id_check: bool = True
    updated_at_trigger: bool = True
    comments: bool = True
    constraint_names: Literal["generated", "postgres"] = "generated"


class EnumDef(Strict):
    enum: str
    description: str
    values: list[str]
