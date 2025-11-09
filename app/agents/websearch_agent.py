"""
Web search agent that collects snippets from a web search provider and
summarizes them into concise bullet points using an LLM.

The module ships a small provider adapter for Tavily when available and
falls back to a stub that returns an empty list (caller should handle it).

Functions:
    - search_web(query, provider='tavily', max_results=5) -> list[str]
    - summarize_web_results(snippets) -> str
    - aggregate_and_summarize(query, max_results=5) -> str

This keeps the integration pluggable and testable. Ensure you set
TAVILY_API_KEY in env vars if using Tavily.
"""

from __future__ import annotations

import os
from typing import List

from app.core.model_loader import load_model
from app.prompts.registry import WEBSEARCH_SUMMARY_PROMPT
from langsmith import traceable


def _tavily_search(query: str, max_results: int = 5) -> List[str]:
    """Attempt to query Tavily and return a list of snippet strings.

    This function tries to import and use the Tavily client. If the
    client or credentials are missing it returns an empty list so the
    caller can handle the absence gracefully.
    """
    try:
        print(f"🔍 Attempting Tavily search for: {query}")
        
        # Check for API key first
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            print("⚠️ TAVILY_API_KEY not found in environment")
            return []
        
        print("   API key found, importing tavily...")
        from tavily import TavilyClient  # Correct import
        
        print("   Creating Tavily client...")
        client = TavilyClient(api_key=api_key)
        
        print(f"   Searching with max_results={max_results}...")
        # Correct Tavily API call
        response = client.search(query=query, max_results=max_results)
        
        print(f"   Response type: {type(response)}")
        print(f"   Response keys: {response.keys() if isinstance(response, dict) else 'N/A'}")
        
        snippets: List[str] = []
        
        # Tavily returns {'results': [...]}
        if isinstance(response, dict) and 'results' in response:
            results = response['results']
            print(f"   Found {len(results)} results")
            
            for i, r in enumerate(results):
                # Tavily results have 'content' field
                content = r.get("content") or r.get("snippet") or r.get("summary")
                if content:
                    snippets.append(str(content))
                    print(f"   Result {i+1}: {str(content)[:100]}...")
        else:
            print(f"   ⚠️ Unexpected response structure: {response}")
            
        print(f"   ✅ Collected {len(snippets)} snippets")
        return snippets
        
    except ImportError as e:
        print(f"❌ Tavily import error: {e}")
        print("   Install with: pip install tavily-python")
        return []
    except Exception as e:
        print(f"❌ Tavily search error: {e}")
        import traceback
        traceback.print_exc()
        return []


@traceable(name="WebSearchAgent.Search")
def search_web(query: str, provider: str = "tavily", max_results: int = 5) -> List[str]:
    """Search the web and return a list of snippet strings.

    provider: currently only 'tavily' is supported as a first-class
    adapter. Unsupported providers return an empty list.
    """
    if provider == "tavily":
        return _tavily_search(query, max_results=max_results)
    # Add other providers here (bing, google) as adapters.
    print(f"⚠️ Unsupported search provider: {provider}")
    return []


@traceable(name="WebSearchAgent.Summarize")
def summarize_web_results(snippets: List[str]) -> str:
    """Summarize web snippets into at most three bullet points.

    Uses the project's model loader to create an LLM with temperature=0.2.
    Returns the raw LLM text output.
    """
    if not snippets:
        print("⚠️ No snippets to summarize")
        return ""

    print(f"📝 Summarizing {len(snippets)} web snippets...")
    
    try:
        llm = load_model()
        
        # Use LCEL syntax instead of LLMChain
        chain = WEBSEARCH_SUMMARY_PROMPT | llm
        
        combined_snippets = "\n\n".join(snippets[:10])
        print(f"   Combined snippets length: {len(combined_snippets)} chars")
        
        # Use invoke instead of run
        raw = chain.invoke({"snippets": combined_snippets})
        
        # Handle different response types
        if hasattr(raw, 'content'):
            summary = str(raw.content)
        else:
            summary = str(raw)
        
        print(f"   ✅ Generated summary: {summary[:200]}...")
        return summary
            
    except Exception as e:
        print(f"❌ Summarization error: {e}")
        import traceback
        traceback.print_exc()
        return ""


@traceable(name="WebSearchAgent.AggregateAndSummarize")
def aggregate_and_summarize(query: str, max_results: int = 5) -> str:
    """High-level helper that searches and summarizes results.

    Returns the summary string (may be empty if no snippets or errors).
    """
    print(f"\n🌐 Web search pipeline for: {query}")
    snippets = search_web(query, max_results=max_results)
    
    if not snippets:
        print("⚠️ No snippets found, returning empty summary")
        return ""
    
    return summarize_web_results(snippets)


__all__ = ["search_web", "summarize_web_results", "aggregate_and_summarize"]