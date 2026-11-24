"""The icon and logo links in index.html and the README must point at real files."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_index_html_icons_exist_in_public():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    hrefs = re.findall(r'<link[^>]*rel="[^"]*icon[^"]*"[^>]*href="(/[^"]+)"', html)

    assert len(hrefs) == 3
    for href in hrefs:
        assert (ROOT / "frontend" / "public" / href.lstrip("/")).is_file(), href


def test_readme_logos_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    header = readme.split("</h1>", 1)[0]
    paths = re.findall(r'(?:src|srcset)="([^"]+)"', header)

    assert len(paths) == 2
    for path in paths:
        assert (ROOT / path).is_file(), path
