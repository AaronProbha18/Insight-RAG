"""Render the model as one self-contained static HTML page (data embedded as JSON, no network)."""
from __future__ import annotations

import datetime
import json
import subprocess
import sys

from model import ROOT, Model, all_columns, load


def payload(m: Model) -> dict:
    tables = []
    for t in m.ordered_tables():
        cols = []
        for c in all_columns(t):
            cols.append({"name": c.name, "type": c.type, "nullable": c.nullable, "pii": c.pii,
                         "default": c.default, "enum": c.enum, "references": c.references,
                         "unique": c.unique, "description": c.description,
                         "system": c.name in ("id", "public_id", "created_at", "updated_at")})
        tables.append({"name": t.table, "domain": t.domain, "status": t.status,
                       "description": t.description, "prefix": t.public_id_prefix,
                       "immutable": t.immutable, "mutable_columns": t.mutable_columns,
                       "columns": cols, "checks": [c.model_dump() for c in t.checks],
                       "unique": t.unique, "indexes": [i.model_dump() for i in t.indexes],
                       "rules": [r.model_dump() for r in t.business_rules],
                       "questions": [q.model_dump() for q in t.open_questions],
                       "samples": t.sample_rows})
    return {
        "domains": [d.model_dump() for d in sorted(m.domains.values(), key=lambda d: d.load_order)],
        "tables": tables,
        "enums": [e.model_dump() for e in sorted(m.enums.values(), key=lambda e: e.enum)],
    }


def version() -> str:
    try:
        v = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True,
                           text=True, timeout=5)
        if v.returncode == 0 and v.stdout.strip():
            return v.stdout.strip()
    except Exception:
        pass
    return "no git"


def main() -> int:
    data = payload(load())
    data["build"] = {"version": version(),
                     "at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%MZ")}
    blob = json.dumps(data).replace("</", "<\\/").replace("<!--", "<\\!--")
    html = (ROOT / "tools" / "explorer.template.html").read_text().replace("/*__DATA__*/null", blob)
    out = ROOT / "build" / "explorer.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    print(f"wrote {out.relative_to(ROOT)} ({len(html) // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
