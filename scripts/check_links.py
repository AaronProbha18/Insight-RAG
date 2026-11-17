"""Fail on broken relative links and #anchors in tracked Markdown files.

Usage: python scripts/check_links.py   (from anywhere inside the repo)
Standard library only, so CI needs no extra dependency.
"""

import re
import subprocess
import sys
from pathlib import Path

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
HEADING = re.compile(r"^#{1,6}\s+(.*)$")
FENCE = re.compile(r"^\s*(```|~~~)")
CODE_SPAN = re.compile(r"`[^`]*`")  # links inside `inline code` are examples, not links


def _prose_lines(text: str):
    """(line number, line) outside fenced code blocks."""
    fenced = False
    for n, line in enumerate(text.splitlines(), 1):
        if FENCE.match(line):
            fenced = not fenced
        elif not fenced:
            yield n, line


def anchors(path: Path) -> set[str]:
    """GitHub-style heading anchors, including -1, -2 suffixes for duplicates."""
    seen: dict[str, int] = {}
    out = set()
    for _, line in _prose_lines(path.read_text(encoding="utf-8")):
        if m := HEADING.match(line):
            slug = re.sub(r"[^\w\- ]", "", m.group(1).strip().lower()).replace(" ", "-")
            count = seen.get(slug, 0)
            seen[slug] = count + 1
            out.add(slug if count == 0 else f"{slug}-{count}")
    return out


def broken_links(files: list[Path]) -> list[str]:
    errors = []
    for md in files:
        for n, line in _prose_lines(md.read_text(encoding="utf-8")):
            for target in LINK.findall(CODE_SPAN.sub("", line)):
                if re.match(r"[a-z]+:", target):  # http:, https:, mailto:
                    continue
                path_part, _, anchor = target.partition("#")
                dest = (md.parent / path_part).resolve() if path_part else md
                if not dest.exists():
                    errors.append(f"{md}:{n}: broken link -> {target}")
                elif anchor and dest.suffix == ".md" and anchor not in anchors(dest):
                    errors.append(f"{md}:{n}: missing anchor -> {target}")
    return errors


def main() -> int:
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    tracked = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    errors = broken_links([root / f for f in tracked])
    print("\n".join(errors) or f"links ok ({len(tracked)} files)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
