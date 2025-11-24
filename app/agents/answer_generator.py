"""
Answer generator for Corrective/Adaptive RAG.

This module constructs a context-aware prompt with named citations based on the
original document filename instead of generic Doc indexes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.core.model_loader import load_model
from app.prompts.registry import ANSWER_PROMPT
from langsmith import traceable


def _build_context(docs: List[Any]) -> Tuple[str, List[str]]:
    """
    Create context using actual document filenames as citation labels.
    Supports either:
      - List[dict] where dict contains page_content + metadata
      - List[str] for fallback plain-text input (very rare)
    """
    parts: List[str] = []
    labels: List[str] = []

    for i, doc in enumerate(docs, start=1):

        # Handle vectorstore dict objects
        if isinstance(doc, dict):
            content = str(doc.get("page_content", "")).strip()[:2000]
            metadata = doc.get("metadata", {}) or {}

            # Correct lookup priority
            source = (
                metadata.get("source_file") or
                metadata.get("source") or
                f"Doc_{i}"
            )

        # Fallback for plain string docs
        else:
            content = str(doc).strip()[:2000]
            source = f"Doc_{i}"

        label = f"[{source}]"
        parts.append(f"{label} {content}")
        labels.append(source)

    return "\n\n".join(parts), labels


@traceable(name="AnswerGenerator.Generate")
def generate_answer(query: str, docs: List[Any], llm: Optional[Any] = None) -> Dict[str, Any]:
    if not docs:
        return {"answer": "I don't know.", "used_docs": []}

    context_str, labels = _build_context(docs)

    if llm is None:
        try:
            llm = load_model(temperature=0.3)
        except Exception:
            return {"answer": "I don't know.", "used_docs": []}

    chain = ANSWER_PROMPT | llm

    try:
        raw = chain.invoke({"context": context_str, "query": query})
        text = raw.content.strip() if hasattr(raw, "content") else str(raw).strip()

        used = []
        for label in labels:
            if f"[{label}]" in text:
                used.append(label)

        if not text or "I don't know" in text:
            return {"answer": "I don't know.", "used_docs": used}

        return {"answer": text, "used_docs": used}

    except Exception:
        return {"answer": "I don't know.", "used_docs": []}


__all__ = ["generate_answer"]
