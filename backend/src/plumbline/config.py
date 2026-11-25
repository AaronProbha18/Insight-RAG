"""Rejects the pre-rename `INSIGHT_RAG_*` environment variables."""

import os
from collections.abc import Mapping

LEGACY_PREFIX = "INSIGHT_RAG_"
NEW_PREFIX = "PLUMBLINE_"


class LegacyConfigError(RuntimeError): ...


def check_legacy_env(environ: Mapping[str, str] = os.environ) -> None:
    """Raise LegacyConfigError naming PLUMBLINE_<X> for each INSIGHT_RAG_<X> set."""
    renames = [
        f"{name} -> {NEW_PREFIX}{name.removeprefix(LEGACY_PREFIX)}"
        for name in sorted(environ)
        if name.startswith(LEGACY_PREFIX)
    ]
    if renames:
        raise LegacyConfigError(
            "Old environment variables are no longer read. Rename: "
            + ", ".join(renames)
        )
