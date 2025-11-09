"""
Document relevance grader used in Corrective RAG.

This module exposes a single function `grade_document` which accepts a
user query and a document snippet (kept under 1000 characters) and returns
a structured decision produced by a deterministic LLM (temperature=0).

If the LLM output cannot be parsed as JSON the module falls back to a
lightweight deterministic heuristic based on token overlap to ensure
reliability and low cost.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from app.core.model_loader import load_model
from app.prompts.registry import DOC_GRADER_PROMPT
from langsmith import traceable


def _extract_json(text: str) -> Optional[str]:
    """Extract the first JSON object substring from model output, if any."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else None


def _heuristic_score(query: str, doc_text: str) -> Dict[str, Any]:
    """Simple deterministic heuristic: token overlap ratio -> score.

    Returns a dict with 'relevant', 'score', and 'reason'.
    """
    # Basic tokenization: alphanumeric words
    q_tokens = set(re.findall(r"\w+", query.lower()))
    d_tokens = set(re.findall(r"\w+", doc_text.lower()))
    if not q_tokens:
        return {"relevant": False, "score": 0.0, "reason": "Empty query"}

    overlap = q_tokens.intersection(d_tokens)
    ratio = len(overlap) / len(q_tokens)

    # Score is ratio, clipped to [0,1]
    score = max(0.0, min(1.0, float(ratio)))
    relevant = score >= 0.3  # Lower threshold from 0.5 to 0.3
    reason = (
        f"Token overlap {len(overlap)}/{len(q_tokens)} (heuristic fallback)"
        if overlap
        else "No overlapping tokens between query and document"
    )
    
    print(f"🔍 Heuristic scoring: overlap={len(overlap)}, q_tokens={len(q_tokens)}, score={score}, relevant={relevant}")
    
    return {"relevant": relevant, "score": round(score, 2), "reason": reason}


@traceable(name="DocumentGrader.Grade")
def grade_document(query: str, doc_text: str, llm: Optional[Any] = None) -> Dict[str, Any]:
    """Grade document relevance to the query.

    Args:
        query: user question string
        doc_text: document snippet (longer text will be truncated to 1000 chars)
        llm: optional LangChain LLM instance; if None, `load_model()` is used

    Returns:
        dict with keys: 'relevant' (bool), 'score' (float 0.0-1.0), 'reason' (str)
    """
    # Ensure snippet is cost-efficient
    snippet = (doc_text or "")[:1000]
    
    print(f"\n📄 Grading document:")
    print(f"   Query: {query}")
    print(f"   Doc preview: {snippet[:150]}...")

    # Prefer provided llm, otherwise attempt to load Gemini via model_loader
    if llm is None:
        try:
            print("   Loading LLM for grading...")
            llm = load_model(temperature=0)
        except Exception as e:
            print(f"   ⚠️ Failed to load LLM: {e}")
            print("   Falling back to heuristic scoring...")
            return _heuristic_score(query, snippet)

    # Use LCEL syntax instead of LLMChain
    chain = DOC_GRADER_PROMPT | llm
    
    try:
        print("   Invoking LLM chain...")
        # Use invoke instead of run
        raw = chain.invoke({"query": query, "doc_text": snippet})
        
        print(f"   Raw LLM response type: {type(raw)}")
        print(f"   Raw LLM response: {str(raw)[:200]}...")

        # Handle different response types
        if hasattr(raw, 'content'):
            text = str(raw.content)
        else:
            text = str(raw)

        # Try to parse as dict first (some models might return structured output)
        if isinstance(raw, dict):
            relevant = bool(raw.get("relevant", False))
            score = float(raw.get("score", 0.0))
            reason = str(raw.get("reason", ""))
            print(f"   ✅ Parsed dict: relevant={relevant}, score={score}")
            return {"relevant": relevant, "score": round(max(0.0, min(1.0, score)), 2), "reason": reason}

        # Try to extract JSON from text response
        json_sub = _extract_json(text)
        if json_sub:
            print(f"   Extracted JSON: {json_sub}")
            try:
                data = json.loads(json_sub)
                relevant = bool(data.get("relevant", False))
                score = float(data.get("score", 0.0))
                reason = str(data.get("reason", ""))
                print(f"   ✅ Parsed JSON: relevant={relevant}, score={score}")
                return {"relevant": relevant, "score": round(max(0.0, min(1.0, score)), 2), "reason": reason}
            except Exception as parse_err:
                print(f"   ⚠️ JSON parse error: {parse_err}")

    except Exception as e:
        print(f"   ❌ LLM invocation error: {e}")
        import traceback
        traceback.print_exc()
        print("   Falling back to heuristic scoring...")
        return _heuristic_score(query, snippet)

    # Final fallback: deterministic heuristic
    print("   ⚠️ Could not parse LLM output, using heuristic...")
    return _heuristic_score(query, snippet)


__all__ = ["grade_document"]