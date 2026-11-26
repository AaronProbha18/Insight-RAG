import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def workcopy(tmp_path):
    """A throwaway copy of the tools and sample model, so tests can edit YAML and run migrate.
    Returns the model folder; the toolkit copy is two levels up."""
    work = tmp_path / "dm"
    shutil.copytree(ROOT / "tools", work / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    model = work / "models" / "sample"
    shutil.copytree(ROOT / "models" / "sample", model, ignore=shutil.ignore_patterns("build"))
    return model


def run_tool(work, script, *args):
    tools = work.parent.parent / "tools"
    env = {**os.environ, "PYTHONPATH": str(tools)}
    return subprocess.run([sys.executable, str(tools / script), *args],
                          capture_output=True, text=True, env=env, cwd=work.parent.parent)
