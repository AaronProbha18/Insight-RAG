"""The pre-rename names may live only in the legacy checks, their tests and history."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
OLD_NAME = re.compile(r"insight[-_]rag", re.IGNORECASE)
ALLOWED_FILES = {
    "backend/src/plumbline/config.py",  # the legacy env check
    "backend/src/plumbline/connectors/registry.py",  # the legacy group warning
    "backend/tests/test_config.py",
    "backend/tests/test_migrations.py",
    "backend/tests/test_old_names.py",
}
ALLOWED_PREFIXES = ("openspec/changes/", "evals/reports/")  # history and baselines


def test_old_names_only_in_allowed_places():
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    hits = []
    for name in filter(None, listed):
        if name in ALLOWED_FILES or name.startswith(ALLOWED_PREFIXES):
            continue
        try:
            text = (ROOT / name).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue  # binary fixtures, or a file deleted but not yet committed
        hits += [
            f"{name}:{n}: {line.strip()}"
            for n, line in enumerate(text.splitlines(), 1)
            if OLD_NAME.search(line)
        ]

    assert not hits, "old names outside the allow-list:\n" + "\n".join(hits)
