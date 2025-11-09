"""
Answer generator for Corrective/Adaptive RAG.

This module constructs a context-aware prompt with numbered citations and
asks a moderately creative LLM (temperature=0.3) to produce a concise,
fact-based answer. If the model fails or is unsure, it returns "I don't know."

API:
    generate_answer(query: str, docs: List[str], llm=None) -> Dict[str, str]

Returned dict contains 'answer' (str) and 'used_docs' (list[int]) indicating which
documents (by index) were cited.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.core.model_loader import load_model
from app.prompts.registry import ANSWER_PROMPT
from langsmith import traceable


def _build_context(docs: List[str]) -> Tuple[str, List[int]]:
    """Create a numbered context string and return indices mapping.

    Each document will be prefixed with [Doc {i}] where i is 1-based index.
    """
    parts: List[str] = []
    indices: List[int] = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"[Doc {i}] {doc.strip()[:2000]}")
        indices.append(i)
    return "\n\n".join(parts), indices


@traceable(name="AnswerGenerator.Generate")
def generate_answer(query: str, docs: List[str], llm: Optional[Any] = None) -> Dict[str, Any]:
    """Generate a concise, cited answer given query and retrieved docs.

    Args:
        query: user question
        docs: list of document strings (short summaries or snippets)
        llm: optional LangChain LLM; if None, `load_model()` is used

    Returns:
        dict with keys: 'answer' (str), 'used_docs' (List[int])
    """
    if not docs:
        return {"answer": "I don't know.", "used_docs": []}

    # Build context with doc numbering
    context_str, indices = _build_context(docs)

    if llm is None:
        try:
            llm = load_model(temperature=0.3)
        except Exception:
            return {"answer": "I don't know.", "used_docs": []}

    # Use LCEL syntax instead of LLMChain
    chain = ANSWER_PROMPT | llm
    
    try:
        # Use invoke instead of run
        raw = chain.invoke({"context": context_str, "query": query})
        
        # Handle different response types (some LLMs return objects, some strings)
        if hasattr(raw, 'content'):
            text = str(raw.content).strip()
        else:
            text = str(raw).strip()

        used = []
        for i in indices:
            if f"[Doc {i}]" in text:
                used.append(i)

        if not text or "I don't know" in text:
            return {"answer": "I don't know.", "used_docs": used}

        return {"answer": text, "used_docs": used}
    except Exception:
        return {"answer": "I don't know.", "used_docs": []}


__all__ = ["generate_answer"]