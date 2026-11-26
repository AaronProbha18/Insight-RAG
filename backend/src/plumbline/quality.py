"""Read the eval runner's summary file and expose an allowlisted view of it."""

import json
from pathlib import Path

GATED = ("correctness", "faithfulness", "abstention")
COUNTS = ("answerable_questions", "unanswerable_questions")
EVAL_COMMAND = "uv run plumbline-eval"
NOT_MEASURED = {"state": "not_measured", "eval_command": EVAL_COMMAND}


def _number(x) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def _connector(c) -> dict:
    """Allowlisted connector entry; KeyError/TypeError/ValueError if malformed."""
    if c["result"] not in ("PASS", "FAIL") or not isinstance(c["failed"], list):
        raise ValueError("bad connector")
    if not all(_number(c["metrics"][m]) for m in GATED):
        raise ValueError("bad metrics")
    entry = {
        "result": c["result"],
        "failed": [m for m in GATED if m in c["failed"]],
        "metrics": {m: c["metrics"][m] for m in GATED},
    }
    # Optional, both or neither: summaries written before the counts existed lack them.
    counts = [c.get(k) for k in COUNTS]
    if all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in counts):
        entry.update(zip(COUNTS, counts, strict=True))
    return entry


def load_quality(path: Path, running: dict) -> dict:
    """`running` = {"llm", "embeddings", "threshold"} of the app now."""
    try:
        s = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            s["result"] not in ("PASS", "FAIL")
            or s["exit_code"] == 2
            or s["sample"] is not None
        ):
            return dict(NOT_MEASURED)
        models = {k: s["models"][k] for k in ("llm", "embeddings", "judge")}
        if not all(isinstance(v, str) for v in models.values()) or not _number(
            s["threshold"]
        ):
            return dict(NOT_MEASURED)
        connectors = {n: _connector(c) for n, c in s["connectors"].items()}
        run_at = s["started_at"]
        if not connectors or not isinstance(run_at, str):
            return dict(NOT_MEASURED)
    # trust boundary: an unreadable or malformed file means "not measured", never a 500
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return dict(NOT_MEASURED)

    differs = {
        "chat": models["llm"] != running["llm"],
        "embeddings": models["embeddings"] != running["embeddings"],
        "threshold": float(s["threshold"]) != float(running["threshold"]),
    }
    diffs = [k for k, v in differs.items() if v]
    return {
        "state": "measured",
        "result": s["result"],
        "run_at": run_at,
        "models": {
            "chat": models["llm"],
            "embeddings": models["embeddings"],
            "judge": models["judge"],
        },
        "threshold": s["threshold"],
        "config_matches": not diffs,
        "config_differences": diffs,
        "connectors": connectors,
    }
