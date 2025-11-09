"""
Graph orchestrator for Adaptive RAG using LangGraph's StateGraph.

This module constructs a small workflow that wires the existing agents
as nodes and defines conditional transitions between them. The graph
implements the sequence:

    router -> retriever -> doc_grader -> answer_generator -> hallucination

With conditional branches to web search and regeneration when needed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict, Optional, Annotated
from operator import add

from langgraph.graph import StateGraph, START, END
from langsmith import traceable

from app.agents import (
    router_agent,
    retriever_agent,
    document_grader,
    answer_generator,
    hallucination_grader,
    websearch_agent,
)


# Define the state schema for proper state persistence
class GraphState(TypedDict, total=False):
    """State schema for the Adaptive RAG graph."""
    query: str
    k: int
    loader: Optional[Any]
    route: str
    route_reason: str
    retrieved_docs: List[str]
    doc_grades: List[Dict[str, Any]]
    avg_doc_score: float
    any_relevant: bool
    answer: str
    used_docs: List[int]
    grounded: bool
    unsupported_claims: List[str]
    web_summary: str
    error: Optional[str]


@traceable(name="Graph.Node.Router")
def _node_router(state: GraphState) -> Dict[str, Any]:
    """Route the query to either vector search or web search."""
    query = state.get("query", "")
    
    if not query:
        print("⚠️ WARNING: Query is empty in router!")
        return {"route": "vector", "route_reason": "Empty query, defaulting to vector"}
    
    decision = router_agent.route_query(query)
    print(f"Router decision: {decision}")
    
    route = decision.get("route", "vector") if decision else "vector"
    reason = decision.get("reason", "") if decision else ""
    
    return {"route": route, "route_reason": reason}


@traceable(name="Graph.Node.Retriever")
def _node_retriever(state: GraphState) -> Dict[str, Any]:
    """Retrieve relevant documents from the vector store."""
    query = state.get("query", "")
    k = int(state.get("k", 4))
    loader = state.get("loader")
    
    print(f"Retrieving docs for query: '{query}', k={k}")
    
    if not query:
        print("⚠️ WARNING: Query is empty in retriever!")
        return {"retrieved_docs": [], "error": "Empty query in retriever"}
    
    try:
        out = retriever_agent.retrieve(query, k=k, loader=loader)
        docs = out.get("retrieved_docs", [])
        
        print(f"Retrieved {len(docs)} documents")
        if docs:
            print(f"First doc preview: {docs[0][:100]}...")
        
        return {"retrieved_docs": docs}
    except Exception as e:
        print(f"❌ Error in retriever: {e}")
        return {"retrieved_docs": [], "error": str(e)}


@traceable(name="Graph.Node.DocGrader")
def _node_doc_grader(state: GraphState) -> Dict[str, Any]:
    """Grade the relevance of retrieved documents."""
    query = state.get("query", "")
    docs: List[str] = state.get("retrieved_docs", [])
    
    if not docs:
        print("⚠️ No documents to grade")
        return {"doc_grades": [], "avg_doc_score": 0.0, "any_relevant": False}
    
    grades = []
    for d in docs:
        g = document_grader.grade_document(query, d)
        grades.append(g)

    # Compute a simple aggregate: average score and any relevant flag
    scores = [g.get("score", 0.0) for g in grades if isinstance(g.get("score", None), (int, float))]
    avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
    any_relevant = any(bool(g.get("relevant", False)) for g in grades)
    
    print(f"Document grading: avg_score={avg_score}, any_relevant={any_relevant}")
    
    return {"doc_grades": grades, "avg_doc_score": avg_score, "any_relevant": any_relevant}


@traceable(name="Graph.Node.AnswerGenerator")
def _node_answer_generator(state: GraphState) -> Dict[str, Any]:
    """Generate an answer based on the query and retrieved documents."""
    query = state.get("query", "")
    docs: List[str] = state.get("retrieved_docs", [])
    
    print(f"Answer generator received {len(docs)} docs for query: '{query}'")
    
    if not query:
        return {"answer": "Error: No query provided.", "used_docs": []}
    
    if not docs:
        return {"answer": "No relevant documents were found to answer your question.", "used_docs": []}
    
    # Prefer relevant docs if grades are present
    grades = state.get("doc_grades", [])
    if grades:
        relevant_docs = [d for d, g in zip(docs, grades) if g.get("relevant", False)]
        docs_to_use = relevant_docs if relevant_docs else docs
        print(f"Using {len(docs_to_use)} relevant docs out of {len(docs)} total")
    else:
        docs_to_use = docs

    try:
        ans = answer_generator.generate_answer(query, docs_to_use)
        answer = ans.get("answer", "I don't know.")
        used_docs = ans.get("used_docs", [])
        
        print(f"Generated answer: {answer[:100]}...")
        
        return {"answer": answer, "used_docs": used_docs}
    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        return {"answer": f"Error generating answer: {str(e)}", "used_docs": []}


@traceable(name="Graph.Node.Hallucination")
def _node_hallucination(state: GraphState) -> Dict[str, Any]:
    """Check if the generated answer is grounded in the retrieved documents."""
    query = state.get("query", "")
    answer = state.get("answer", "")
    docs: List[str] = state.get("retrieved_docs", [])
    
    if not answer or answer.startswith("Error:") or answer == "I don't know.":
        # Skip hallucination check for error/empty answers
        return {"grounded": True, "unsupported_claims": []}
    
    try:
        res = hallucination_grader.grade_hallucination(query, answer, docs)
        grounded = bool(res.get("grounded", False))
        unsupported = res.get("unsupported_claims", [])
        
        print(f"Hallucination check: grounded={grounded}, unsupported_claims={len(unsupported)}")
        
        return {"grounded": grounded, "unsupported_claims": unsupported}
    except Exception as e:
        print(f"❌ Error in hallucination grader: {e}")
        return {"grounded": True, "unsupported_claims": []}


@traceable(name="Graph.Node.Regenerate")
def _node_regenerate(state: GraphState) -> Dict[str, Any]:
    """Regenerate the answer to fix hallucinations."""
    query = state.get("query", "")
    docs: List[str] = state.get("retrieved_docs", [])
    
    print("Regenerating answer to fix hallucinations...")
    
    try:
        new_answer = hallucination_grader.regenerate_answer(query, docs)
        return {"answer": new_answer}
    except Exception as e:
        print(f"❌ Error regenerating answer: {e}")
        return {"answer": "Error: Could not regenerate answer."}


@traceable(name="Graph.Node.WebSearch")
def _node_web_search(state: GraphState) -> Dict[str, Any]:
    """Perform web search to gather additional information."""
    query = state.get("query", "")
    
    print(f"Performing web search for query: '{query}'")
    
    if not query:
        return {"retrieved_docs": [], "web_summary": ""}
    
    try:
        summary = websearch_agent.aggregate_and_summarize(query)
        # Treat the web summary as a single retrieved doc for downstream components
        docs = [summary] if summary else []
        
        print(f"Web search returned summary of length: {len(summary) if summary else 0}")
        
        return {"retrieved_docs": docs, "web_summary": summary}
    except Exception as e:
        print(f"❌ Error in web search: {e}")
        return {"retrieved_docs": [], "web_summary": "", "error": str(e)}


# Conditional edge functions
def route_after_router(state: GraphState) -> Literal["retriever", "web_search"]:
    """Decide whether to go to retriever or web_search based on router decision."""
    route = state.get("route", "vector")
    print(f"Routing after router: {route}")
    return "retriever" if route == "vector" else "web_search"


def route_after_doc_grader(state: GraphState) -> Literal["answer_generator", "web_search"]:
    """Decide whether docs are relevant enough or need web search."""
    any_relevant = state.get("any_relevant", False)
    avg_score = state.get("avg_doc_score", 0.0)
    
    if any_relevant or avg_score >= 0.5:
        print("Routing to answer_generator (docs are relevant)")
        return "answer_generator"
    else:
        print("Routing to web_search (docs not relevant enough)")
        return "web_search"


def route_after_hallucination(state: GraphState) -> Literal["regenerate", "__end__"]:
    """Decide whether to regenerate answer or end."""
    grounded = state.get("grounded", True)
    
    if not grounded:
        print("Routing to regenerate (hallucination detected)")
        return "regenerate"
    else:
        print("Routing to END (answer is grounded)")
        return "__end__"


@traceable(name="GraphOrchestrator.BuildGraph")
def build_graph() -> Any:
    """Build and compile the LangGraph StateGraph for Adaptive RAG.

    Returns:
        A compiled StateGraph instance ready for invoke/stream calls.
    """
    # Use the typed state schema for proper state persistence
    builder = StateGraph(GraphState)

    # Add nodes (name -> callable)
    builder.add_node("router", _node_router)
    builder.add_node("retriever", _node_retriever)
    builder.add_node("doc_grader", _node_doc_grader)
    builder.add_node("answer_generator", _node_answer_generator)
    builder.add_node("hallucination", _node_hallucination)
    builder.add_node("regenerate", _node_regenerate)
    builder.add_node("web_search", _node_web_search)

    # Entry point
    builder.add_edge(START, "router")

    # Conditional edges using add_conditional_edges
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retriever": "retriever",
            "web_search": "web_search"
        }
    )

    # Normal vector path
    builder.add_edge("retriever", "doc_grader")

    # Conditional: if no relevant docs, go to web search, else generate answer
    builder.add_conditional_edges(
        "doc_grader",
        route_after_doc_grader,
        {
            "answer_generator": "answer_generator",
            "web_search": "web_search"
        }
    )

    # Web search flows into answer generation
    builder.add_edge("web_search", "answer_generator")

    # Answer -> hallucination grader
    builder.add_edge("answer_generator", "hallucination")

    # Conditional: if hallucination detected -> regenerate, else end
    builder.add_conditional_edges(
        "hallucination",
        route_after_hallucination,
        {
            "regenerate": "regenerate",
            "__end__": END
        }
    )

    # After regeneration, check hallucination again
    builder.add_edge("regenerate", "hallucination")

    graph = builder.compile()
    return graph


@traceable(name="GraphOrchestrator.RunGraph")
def run_graph(query: str, k: int = 4, loader: Any = None) -> Dict[str, Any]:
    """Helper that builds the graph, invokes it with the input state, and
    returns the final state dictionary.
    
    Args:
        query: The user's query
        k: Number of documents to retrieve
        loader: The vector store loader instance
        
    Returns:
        Final state dictionary with the answer and metadata
    """
    graph = build_graph()
    initial_state: GraphState = {
        "query": query,
        "k": k,
        "loader": loader
    }
    
    print(f"\n{'='*60}")
    print(f"Starting RAG pipeline for query: '{query}'")
    print(f"{'='*60}\n")

    # Invoke the graph
    result = graph.invoke(initial_state)
    
    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"{'='*60}\n")

    return result


__all__ = ["build_graph", "run_graph"]