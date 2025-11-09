"""
Routing agent to decide whether a query should be answered from the
internal vectorstore or via a web search.

The agent uses a deterministic LLM (Gemini via LangChain) with
temperature=0 and returns a JSON-like decision. If the model output
is not valid JSON the module falls back to a small deterministic
keyword-based heuristic to ensure robustness.

API:
    route = route_query(query: str, llm=None) -> dict

The returned dict has keys:
    - route: "vector" or "web"
    - reason: brief explanation
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from app.core.model_loader import load_model
from app.prompts.registry import ROUTER_PROMPT
from langsmith import traceable


def _extract_json(text: str) -> Optional[str]:
    """Try to extract a JSON object substring from the model output."""
    # Simple regex to find {...} block
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else None


def _heuristic_route(query: str) -> Dict[str, str]:
    """Fallback deterministic heuristic used when LLM output is unusable.

    Looks for tokens indicating time-sensitivity or current events.
    """
    q = query.lower()
    time_keywords = (
        "today",
        "yesterday",
        "tomorrow",
        "current",
        "latest",
        "news",
        "trend",
        "trending",
        "202",
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
        "2025",
        "when",
        "what year",
        "price",
        "stock",
        "weather",
        "breaking",
    )
    for kw in time_keywords:
        if kw in q:
            return {"route": "web", "reason": f"Detected time-sensitive keyword '{kw}'"}

    return {"route": "vector", "reason": "Query appears factual/document-centered"}


@traceable(name="RouterAgent.RouteQuery")
def route_query(query: str, llm: Optional[Any] = None) -> Dict[str, str]:
    """Decide routing for the given query.

    Args:
        query: user query string
        llm: optional LangChain LLM object; if None, `load_model()` is used

    Returns:
        dict with keys 'route' ("vector" or "web") and 'reason' (brief string)
    """
    if llm is None:
        try:
            llm = load_model()
        except Exception:
            # If the LLM cannot be loaded, fall back to heuristic
            return _heuristic_route(query)

    # Use LCEL syntax instead of LLMChain
    chain = ROUTER_PROMPT | llm
    
    try:
        # Use invoke instead of run
        raw = chain.invoke({"query": query})
    except Exception:
        return _heuristic_route(query)

    # Handle different response types
    if hasattr(raw, 'content'):
        text = str(raw.content)
    else:
        text = str(raw)

    # Try to parse as dict first (some models might return structured output)
    if isinstance(raw, dict):
        route = raw.get("route")
        reason = raw.get("reason", "")
        if route in ("vector", "web"):
            return {"route": route, "reason": reason}

    # Try to extract JSON from output
    json_sub = _extract_json(text)
    if json_sub:
        try:
            data = json.loads(json_sub)
            route = data.get("route")
            reason = data.get("reason", "")
            if route in ("vector", "web"):
                return {"route": route, "reason": reason}
        except Exception:
            # fall through to heuristic
            pass

    # Last resort: deterministic heuristic
    return _heuristic_route(query)


__all__ = ["route_query"]