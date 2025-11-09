"""
Hallucination grader for Self-RAG.

This module compares a generated answer against the retrieved context
and detects unsupported claims. It prefers a deterministic LLM check
(temperature=0) that returns strict JSON. If the LLM fails or output
is not parseable, a lightweight heuristic marks sentences unsupported
when they lack sufficient token overlap with the context.

API:
    grade_hallucination(query: str, answer: str, docs: List[str], llm=None) -> dict

Returns:
    {"grounded": bool, "unsupported_claims": List[str]}
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from app.core.model_loader import load_model
from app.prompts.registry import HALLUCINATION_PROMPT, REGENERATE_PROMPT
from langsmith import traceable


def _extract_json(text: str) -> Optional[str]:
    """Extract the first JSON object substring from model output, if any."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else None


def _heuristic_unsupported(answer: str, context: str) -> List[str]:
    """Heuristic detection of unsupported answer sentences.

    Splits the answer into candidate sentences and labels a sentence
    unsupported when fewer than 50% of its content words appear in
    the context.
    """
    # Normalize and tokenize
    ctx_tokens = set(re.findall(r"\w+", context.lower()))
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    unsupported: List[str] = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        tokens = [t for t in re.findall(r"\w+", sent.lower()) if len(t) > 2]
        if not tokens:
            continue
        overlap = sum(1 for t in tokens if t in ctx_tokens)
        ratio = overlap / len(tokens)
        # If less than half tokens overlap, mark as unsupported
        if ratio < 0.5:
            # Keep short claim for reporting
            claim = sent if len(sent) <= 200 else sent[:197] + "..."
            unsupported.append(claim)

    return unsupported


@traceable(name="HallucinationGrader.Grade")
def grade_hallucination(query: str, answer: str, docs: List[str], llm: Optional[Any] = None) -> Dict[str, Any]:
    """Grade whether the answer is grounded in the provided documents.

    Args:
        query: original user query
        answer: generated answer text
        docs: list of context document strings
        llm: optional LangChain LLM; if None, `load_model()` is used

    Returns:
        dict with keys 'grounded' (bool) and 'unsupported_claims' (List[str])
    """
    context = "\n\n".join(docs)

    if llm is None:
        try:
            llm = load_model()
        except Exception:
            # If LLM unavailable, use heuristic
            unsupported = _heuristic_unsupported(answer, context)
            return {"grounded": len(unsupported) == 0, "unsupported_claims": unsupported}

    # Use LCEL syntax instead of LLMChain
    chain = HALLUCINATION_PROMPT | llm
    
    try:
        # Use invoke instead of run
        raw = chain.invoke({"query": query, "answer": answer, "context": context})

        # Handle different response types
        if hasattr(raw, 'content'):
            text = str(raw.content)
        else:
            text = str(raw)

        # Try to parse as dict first
        if isinstance(raw, dict):
            grounded = bool(raw.get("grounded", False))
            unsupported = list(raw.get("unsupported_claims", []))
            return {"grounded": grounded, "unsupported_claims": unsupported}

        # Try to extract JSON from text response
        json_sub = _extract_json(text)
        if json_sub:
            try:
                data = json.loads(json_sub)
                grounded = bool(data.get("grounded", False))
                unsupported = list(data.get("unsupported_claims", []))
                return {"grounded": grounded, "unsupported_claims": unsupported}
            except Exception:
                pass

    except Exception:
        unsupported = _heuristic_unsupported(answer, context)
        return {"grounded": len(unsupported) == 0, "unsupported_claims": unsupported}

    unsupported = _heuristic_unsupported(answer, context)
    return {"grounded": len(unsupported) == 0, "unsupported_claims": unsupported}


@traceable(name="HallucinationGrader.Regenerate")
def regenerate_answer(query: str, docs: List[str], llm: Optional[Any] = None) -> str:
    """Regenerate an answer strictly grounded on the supplied documents.

    Uses a deterministic LLM call (temperature=0). Returns a short
    grounded answer string or "I don't know." when generation fails.
    """
    context = "\n\n".join(docs)

    if llm is None:
        try:
            llm = load_model()
        except Exception:
            return "I don't know."

    # Use LCEL syntax instead of LLMChain
    chain = REGENERATE_PROMPT | llm
    
    try:
        # Use invoke instead of run
        raw = chain.invoke({"context": context, "query": query})
        
        # Handle different response types
        if hasattr(raw, 'content'):
            text = str(raw.content).strip()
        else:
            text = str(raw).strip()
            
        if not text:
            return "I don't know."
        return text
    except Exception:
        return "I don't know."


__all__ = ["grade_hallucination", "regenerate_answer"]