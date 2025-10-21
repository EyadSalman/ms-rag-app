# backend/rag_pipeline/graph.py
from langgraph.graph import StateGraph, END
from backend.rag_pipeline.agents import mri_agent, research_agent

def route_query(state):
    q = state["query"].lower()
    if any(word in q for word in ["scan", "mri", "lesion", "image", "patient"]):
        agent_type = "mri"
    else:
        agent_type = "research"
    return {"query": q, "agent_type": agent_type}

def process_mri(state):
    return {"answer": mri_agent(state["query"]), "agent_type": "mri"}

def process_research(state):
    return {"answer": research_agent(state["query"]), "agent_type": "research"}

def build_ms_graph():
    g = StateGraph(dict)
    g.add_node("router", route_query)
    g.add_node("mri_agent", process_mri)
    g.add_node("research_agent", process_research)
    g.set_entry_point("router")

    g.add_conditional_edges("router",
        lambda s: s["agent_type"],
        {"mri": "mri_agent", "research": "research_agent"}
    )
    g.add_edge("mri_agent", END)
    g.add_edge("research_agent", END)

    return g.compile()
