"""Enforce the boundary rules in docs/architecture/overview.md."""

import ast
from pathlib import Path

BACKEND = Path(__file__).parents[1]
PKG = BACKEND / "src" / "plumbline"
CONFIG_READERS = {
    "models.py",
    "db.py",
    "api/app.py",
    "evals/runner.py",
}


def _imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_graph_knows_nothing_about_http_or_db():
    imports = _imports(PKG / "graph.py")

    assert not {m for m in imports if m.split(".")[0] in {"psycopg", "fastapi"}}


def test_builtin_connectors_use_only_the_public_contract():
    for path in (PKG / "connectors" / "builtin").glob("*.py"):
        internal = {m for m in _imports(path) if m.startswith("plumbline")}
        assert internal <= {"plumbline.connectors.contract"}, path.name


def test_only_config_owners_read_the_environment():
    readers = {
        p.relative_to(PKG).as_posix()
        for p in PKG.rglob("*.py")
        if "os.environ" in p.read_text(encoding="utf-8")
        or "os.getenv" in p.read_text(encoding="utf-8")
    }

    assert readers <= CONFIG_READERS, readers - CONFIG_READERS
