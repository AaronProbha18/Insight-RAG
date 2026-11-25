"""Generate the PDF and DOCX fixture corpus from evals/corpus-src/*.md.

Run from the repo root:
    uv run --no-project --with reportlab --with python-docx python evals/generate_corpus.py

Supported Markdown subset: #/##/### headings, paragraphs, "- " bullets,
pipe tables, and <!-- pagebreak --> (forces a PDF page break).
"""

import datetime
import io
import zipfile
from pathlib import Path

import docx
from reportlab import rl_config
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    ListFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
)

rl_config.invariant = 1  # no timestamps/random ids, so output is reproducible
FIXED_DATE = datetime.datetime(2026, 1, 1)
ROOT = Path(__file__).parent
SRC = ROOT / "corpus-src"


def blocks(md: str):
    """Yield (kind, payload) blocks: heading, para, bullets, table, pagebreak."""
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
        elif line == "<!-- pagebreak -->":
            yield "pagebreak", None
            i += 1
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            yield "heading", (level, line[level:].strip())
            i += 1
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(lines[i][2:].strip())
                i += 1
            yield "bullets", items
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not set("".join(cells)) <= set("-: "):  # skip the --- separator row
                    rows.append(cells)
                i += 1
            yield "table", rows
        else:
            yield "para", line
            i += 1


def to_pdf(md: str, out: Path) -> None:
    st = getSampleStyleSheet()
    story = []
    for kind, p in blocks(md):
        if kind == "heading":
            story.append(Paragraph(p[1], st[f"Heading{p[0]}"]))
        elif kind == "para":
            story.append(Paragraph(p, st["Normal"]))
            story.append(Spacer(0, 6))
        elif kind == "bullets":
            story.append(
                ListFlowable(
                    [Paragraph(t, st["Normal"]) for t in p], bulletType="bullet"
                )
            )
        elif kind == "table":
            story.append(Table(p))
            story.append(Spacer(0, 6))
        elif kind == "pagebreak":
            story.append(PageBreak())
    title = md.splitlines()[0].lstrip("# ")
    SimpleDocTemplate(str(out), title=title, author="plumbline fixtures").build(story)


def to_docx(md: str, out: Path) -> None:
    d = docx.Document()
    for kind, p in blocks(md):
        if kind == "heading":
            d.add_heading(p[1], p[0])
        elif kind == "para":
            d.add_paragraph(p)
        elif kind == "bullets":
            for t in p:
                d.add_paragraph(t, style="List Bullet")
        elif kind == "table":
            t = d.add_table(rows=len(p), cols=len(p[0]))
            for r, row in enumerate(p):
                for c, text in enumerate(row):
                    t.cell(r, c).text = text
        elif kind == "pagebreak":
            d.add_page_break()
    props = d.core_properties
    props.title = md.splitlines()[0].lstrip("# ")
    props.author = "plumbline fixtures"
    props.created = props.modified = FIXED_DATE
    props.last_modified_by = "plumbline fixtures"
    props.revision = 1
    buf = io.BytesIO()
    d.save(buf)
    # python-docx stamps zip entries with the current time; pin them.
    with (
        zipfile.ZipFile(buf) as src,
        zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst,
    ):
        for info in src.infolist():
            dst.writestr(
                zipfile.ZipInfo(info.filename, FIXED_DATE.timetuple()[:6]),
                src.read(info),
                compress_type=zipfile.ZIP_DEFLATED,
            )


def main() -> None:
    for kind in ("pdf", "docx"):
        (ROOT / "connectors" / kind / "corpus").mkdir(parents=True, exist_ok=True)
    for src in sorted(SRC.glob("*.md")):
        md = src.read_text(encoding="utf-8")
        to_pdf(md, ROOT / "connectors" / "pdf" / "corpus" / f"{src.stem}.pdf")
        to_docx(md, ROOT / "connectors" / "docx" / "corpus" / f"{src.stem}.docx")
        print("generated", src.stem)


if __name__ == "__main__":
    main()
