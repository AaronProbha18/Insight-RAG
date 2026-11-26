"""Render the model as one self-contained static HTML page (data embedded as JSON, no network)."""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys

from model import DEFAULT_MODEL, ROOT, Model, all_columns, column_type, load, model_dir


def payload(m: Model) -> dict:
    tables = []
    for t in m.ordered_tables():
        cols = []
        for c in all_columns(t):
            cols.append({"name": c.name, "type": column_type(c), "nullable": c.nullable, "pii": c.pii,
                         "default": c.default, "references": c.references,
                         "unique": c.unique, "description": c.description,
                         "system": c.name in ("id", "public_id", "created_at", "updated_at")})
        tables.append({"name": t.table, "domain": t.domain, "status": t.status,
                       "description": t.description, "prefix": t.public_id_prefix,
                       "columns": cols, "checks": [c.model_dump() for c in t.checks],
                       "unique": t.unique, "indexes": [i.model_dump() for i in t.indexes],
                       "questions": [q.model_dump() for q in t.open_questions]})
    return {
        "domains": [d.model_dump() for d in sorted(m.domains.values(), key=lambda d: d.load_order)],
        "tables": tables,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    md = model_dir(ap.parse_args().model)
    data = payload(load(md))
    data["build"] = {"version": version(),
                     "at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%MZ")}
    blob = json.dumps(data).replace("</", "<\\/").replace("<!--", "<\\!--")
    html = (ROOT / "tools" / "explorer.template.html").read_text().replace("/*__DATA__*/null", blob)
    out = md / "build" / "explorer.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({len(html) // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
