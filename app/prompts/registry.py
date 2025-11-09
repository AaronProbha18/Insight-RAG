"""
Centralized Prompt Registry

Contains all reusable PromptTemplates for Adaptive, Corrective, and Self-RAG agents.
Do NOT change logic, imports, or function signatures — only refactor prompt storage and usage.
"""

from langchain_core.prompts import PromptTemplate

ROUTER_PROMPT = PromptTemplate.from_template(
    """
You are a routing agent that decides whether a user's question
requires internal knowledge (vectorstore) or external information (web search).

Rules:
- "vector" -> for factual or internal topics.
- "web" -> for real-time or news queries.

Output JSON:
{{
    "route": "vector" or "web",
    "reason": "brief reason"
}}

User query: {query}
"""
)


DOC_GRADER_PROMPT = PromptTemplate.from_template(
    """
You are a document relevance grader. Given a user query and a document, decide whether
the document helps answer the question. Respond ONLY with strict JSON and nothing else.

Output JSON schema (exact keys):
{{
    "relevant": true|false,
    "score": 0.0-1.0,
    "reason": "short explanation (one sentence)"
}}

Use the following few-shot examples to guide formatting and scoring. Examples must be
followed strictly (no surrounding text, no markdown):

# Example 1 (relevant)
Query: How do I reset my password in AcmeApp?
Document: To reset your password in AcmeApp go to Settings -> Account -> Reset Password and follow the on-screen steps.
Response: {{"relevant": true, "score": 0.92, "reason": "Describes the exact reset password flow in AcmeApp."}}

# Example 2 (not relevant)
Query: What is the refund policy for orders placed in 2024?
Document: Our blog post about product design trends in 2021 does not mention refunds or policies.
Response: {{"relevant": false, "score": 0.10, "reason": "Document discusses product design trends, not refunds."}}

Now evaluate the user input below. Return only the single JSON object.

Question: {query}
Document:
{doc_text}
"""
)


ANSWER_PROMPT = PromptTemplate.from_template(
    """
You are a helpful assistant that answers user questions using only the provided context.
Rules:
- Use only facts from the context.
- Cite document indices using [Doc X].
- If unsure, say "I don’t know."
Context:
{context}
Question: {query}
Answer (with citations):
"""
)


HALLUCINATION_PROMPT = PromptTemplate.from_template(
    """
You are a fact-checking assistant.
Compare the answer to the given documents and detect unsupported claims.
Return JSON with keys: 'grounded' (true/false) and 'unsupported_claims' (list of short strings).
Question: {query}
Answer: {answer}
Context Documents:
{context}
"""
)


WEBSEARCH_SUMMARY_PROMPT = PromptTemplate.from_template(
    """
You are a factual web summarizer. Based on the following web snippets, produce 3 concise bullet points
summarizing the key facts.

Web snippets:
{snippets}

Requirements:
- Output exactly 3 bullet points (one per line, prefixed with "- ").
- Be factual and avoid speculation or interpretation beyond the snippets.
- Keep the entire summary between 100 and 150 words total.

Summary (3 bullet points):
-
-
-
"""
)


REGENERATE_PROMPT = PromptTemplate.from_template(
    """
The previous answer contained unsupported claims.
Regenerate a new answer strictly using only the context.
Context:
{context}
Question:
{query}
Grounded answer:
"""
)


__all__ = [
    "ROUTER_PROMPT",
    "DOC_GRADER_PROMPT",
    "ANSWER_PROMPT",
    "HALLUCINATION_PROMPT",
    "WEBSEARCH_SUMMARY_PROMPT",
    "REGENERATE_PROMPT",
]
