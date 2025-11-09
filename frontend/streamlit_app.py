import streamlit as st
import requests
from typing import Dict, Any

# ==========================================================
# 🔧 App Setup
# ==========================================================
st.set_page_config(page_title="InsightRAG", page_icon="🚀", layout="wide")

BASE_URL = "http://127.0.0.1:8000/api/v1"

st.markdown("""
# 🚀 **InsightRAG**
#### _An Adaptive RAG Chat Assistant powered by LangGraph + LangSmith_
""")

st.caption("Upload, query, and converse with your Adaptive RAG pipeline — all from one simple, elegant interface.")

# ==========================================================
# 🌐 Helper Functions
# ==========================================================

def call_api(endpoint: str, payload: Dict[str, Any] = None, files=None):
    """Safely call FastAPI backend."""
    try:
        if files:
            response = requests.post(f"{BASE_URL}/{endpoint}", files=files)
        else:
            response = requests.post(f"{BASE_URL}/{endpoint}", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# ==========================================================
# 🧭 Tabs Layout
# ==========================================================
tab1, tab2, tab3 = st.tabs(["💾 Vector Store", "💬 Query Operations", "🧠 RAG Chatbot"])

# ==========================================================
# 💾 TAB 1: Vector Store Operations
# ==========================================================
with tab1:
    st.subheader("📤 Upload Documents to Vector Store")

    uploaded_files = st.file_uploader(
        "Upload one or more .txt or .json files to build the FAISS index:",
        accept_multiple_files=True,
        type=["txt", "json"]
    )

    if st.button("🚀 Load into FAISS Index"):
        if uploaded_files:
            files = [("files", (f.name, f, f.type)) for f in uploaded_files]
            with st.spinner("Embedding and indexing documents..."):
                result = call_api("load_vectors", files=files)
            if "error" not in result:
                st.success("✅ Vector store updated successfully!")
                st.session_state["faiss_stats"] = result
            else:
                st.error(result["error"])
        else:
            st.warning("Please upload at least one file first.")

    # --- Dashboard Section ---
    st.markdown("---")
    st.subheader("📊 FAISS Index Dashboard")

    if "faiss_stats" in st.session_state:
        stats = st.session_state["faiss_stats"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents Indexed", stats.get("count", 0))
        with col2:
            st.metric("Status", stats.get("status", "unknown").capitalize())
        with col3:
            st.metric("Embedding Model", "sentence-transformers/all-MiniLM-L6-v2")
    else:
        st.info("ℹ️ Load some documents first to view FAISS stats.")

    # --- Query FAISS Section ---
    st.markdown("---")
    st.subheader("🔎 Query the Vector Store")
    query_text = st.text_input("Enter a semantic search query:")
    top_k = st.slider("Number of results (k):", 1, 10, 3)

    if st.button("🔍 Query Vectors"):
        if query_text.strip():
            with st.spinner("Searching FAISS index..."):
                result = call_api("query_vectors", {"query": query_text, "k": top_k})
            if "results" in result:
                st.success(f"✅ Found {len(result['results'])} matching documents.")
                st.json(result)
            else:
                st.error(result.get("error", "Unknown error occurred."))
        else:
            st.warning("Please enter a query first.")

# ==========================================================
# 💬 TAB 2: Direct Model Interaction
# ==========================================================
with tab2:
    st.subheader("🗣️ Ask the Model Directly (LLM)")

    user_prompt = st.text_area(
        "Prompt the Gemini model directly:",
        height=120,
        placeholder="e.g. Explain LangGraph orchestration...",
    )

    if st.button("✨ Generate with Model"):
        if user_prompt.strip():
            with st.spinner("Contacting Gemini model..."):
                result = call_api("model_query", {"prompt": user_prompt})
            if "response" in result:
                st.markdown("### 💡 Model Response")
                st.info(result["response"])
            else:
                st.error(result.get("error", "Unknown error occurred"))
        else:
            st.warning("Please enter a prompt first.")

# ==========================================================
# 🧠 TAB 3: Full RAG Chatbot
# ==========================================================
with tab3:
    st.subheader("💬 Adaptive RAG Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("metadata"):
                with st.expander("🧩 RAG Metadata"):
                    st.json(msg["metadata"])

    user_input = st.chat_input("Ask a question to the RAG pipeline...")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("🧠 Running Adaptive RAG pipeline..."):
            response = call_api("rag_query", {"query": user_input, "k": 3})

        if "error" in response:
            answer = f"❌ Error: {response['error']}"
            metadata = {}
        else:
            answer = response.get("answer") or response.get("response", "⚠️ No answer returned.")
            metadata = {k: v for k, v in response.items() if k not in ["answer", "response"]}

        with st.chat_message("assistant"):
            st.markdown(answer)
            if metadata:
                with st.expander("🧠 RAG Reasoning & Metadata"):
                    st.json(metadata)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "metadata": metadata}
        )

# ==========================================================
# 🎨 Footer
# ==========================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'> <b>InsightRAG</b> — Powered by LangGraph, LangSmith, and FastAPI</p>",
    unsafe_allow_html=True
)
