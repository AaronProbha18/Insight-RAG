import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def workcopy(tmp_path):
    """A throwaway copy of the model and tools, so tests can edit YAML and run migrate."""
    work = tmp_path / "dm"
    for d in ("domains", "enums", "tools", "migrations"):
        shutil.copytree(ROOT / d, work / d, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "model.lock.json", work)
    return work


def run_tool(work, script, *args):
    env = {**os.environ, "PYTHONPATH": str(work / "tools")}
    return subprocess.run([sys.executable, str(work / "tools" / script), *args],
                          capture_output=True, text=True, env=env, cwd=work)
