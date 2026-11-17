import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "check_links.py"
spec = importlib.util.spec_from_file_location("check_links", SCRIPT)
check_links = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_links)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_good_link_and_anchor_pass(tmp_path):
    _write(tmp_path, "b.md", "# Title\n\n## 1. API conventions\n")
    a = _write(
        tmp_path,
        "a.md",
        "[ok](b.md) [anchor](b.md#1-api-conventions) [self](#top)\n# Top\n",
    )

    assert check_links.broken_links([a]) == []


def test_missing_file_and_missing_anchor_fail(tmp_path):
    _write(tmp_path, "b.md", "# Title\n")
    a = _write(tmp_path, "a.md", "[gone](nope.md)\n[bad](b.md#no-such-heading)\n")

    errors = check_links.broken_links([a])

    assert len(errors) == 2
    assert "broken link -> nope.md" in errors[0]
    assert "missing anchor -> b.md#no-such-heading" in errors[1]


def test_code_and_external_links_are_ignored(tmp_path):
    a = _write(
        tmp_path,
        "a.md",
        "```\n[x](fenced.md)\n```\nInline `[x](span.md)` and [web](https://example.com)\n",
    )

    assert check_links.broken_links([a]) == []


def test_duplicate_headings_get_numbered_anchors(tmp_path):
    b = _write(tmp_path, "b.md", "## Setup\n## Setup\n")

    assert {"setup", "setup-1"} <= check_links.anchors(b)


def test_repository_has_no_broken_links():
    assert check_links.main() == 0
