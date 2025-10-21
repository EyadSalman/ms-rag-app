# backend/rag_pipeline/__init__.py

from backend.rag_pipeline.graph import build_ms_graph

# 🧠 build LangGraph once at startup
_graph = None

def get_gemini_response(query: str):
    """
    Called by FastAPI route /ask_gemini/
    Uses the LangGraph workflow to handle the query.
    """
    global _graph
    if _graph is None:
        _graph = build_ms_graph()

    try:
        result = _graph.invoke({"query": query})
        print(f"RAG Pipeline result: {result}")
        return result.get("answer", "No response generated.")
    except Exception as e:
        return f"Error in RAG pipeline: {e}"
