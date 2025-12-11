# backend/rag_pipeline/agents.py

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import google.generativeai as genai

# ===========================
# 🔐 GOOGLE API CONFIG
# ===========================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("⚠️ WARNING: GOOGLE_API_KEY missing — LLM disabled.")

# ===========================
# 🔧 LAZY IMPORTS
# ===========================
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ===========================
# 🔁 Lazy Instances
# ===========================
_embedding = None
_llm = None
_MRI_DB = None
_RESEARCH_DB = None


# ===========================
# 🔌 Embedding Loader
# ===========================
def get_embedding():
    global _embedding
    if _embedding is None:
        print("🔄 Loading embedding model...")
        _embedding = HuggingFaceEmbeddings(
            model_name="mixedbread-ai/mxbai-embed-large-v1"
        )
    return _embedding


# ===========================
# 🤖 LLM Loader
# ===========================
def get_llm():
    global _llm
    if _llm is None:
        print("🔄 Loading Gemini Flash...")
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", temperature=0.1
        )
    return _llm


# ===========================
# 📁 MRI Vectorstore Loader
# ===========================
def get_mri_db():
    global _MRI_DB
    if _MRI_DB is None:
        path = os.path.join("backend", "vectorstores", "mri_db_free")
        if not os.path.exists(path):
            print("⚠️ MRI vectorstore missing:", path)
            return None

        print("📁 Loading MRI vectorstore...")
        _MRI_DB = Chroma(
            persist_directory=path,
            embedding_function=get_embedding()
        )
    return _MRI_DB


# ===========================
# 📚 Research Vectorstore Loader
# ===========================
def get_research_db():
    global _RESEARCH_DB
    if _RESEARCH_DB is None:
        path = os.path.join("backend", "vectorstores", "research_db_free")
        if not os.path.exists(path):
            print("⚠️ Research vectorstore missing:", path)
            return None

        print("📁 Loading Research vectorstore...")
        _RESEARCH_DB = Chroma(
            persist_directory=path,
            embedding_function=get_embedding()
        )
    return _RESEARCH_DB


# ===========================
# 🧠 MRI RAG Agent
# ===========================
def mri_agent(query: str):
    db = get_mri_db()
    llm = get_llm()

    if db is None:
        return {"answer": "MRI knowledge base unavailable.", "agent_type": "mri"}

    results = db.similarity_search(query, k=3)
    context = "\n".join(r.page_content for r in results) or "No MRI data available."

    prompt = f"""
You are an MRI interpretation assistant.

MRI Context:
{context}

Question:
{query}

Answer:
"""

    resp = llm.invoke(prompt)
    return {
        "answer": resp.content,
        "agent_type": "mri",
        "sources": [r.metadata.get("source") for r in results],
    }


# ===========================
# 📚 Research RAG Agent
# ===========================
def research_agent(query: str):
    db = get_research_db()
    llm = get_llm()

    if db is None:
        return "MS research database unavailable."

    results = db.similarity_search(query, k=3)
    context = "\n".join(r.page_content for r in results) or "No research found."

    sources = []
    seen = set()
    for r in results:
        src = r.metadata.get("source")
        if src and src not in seen:
            sources.append(src)
            seen.add(src)

    prompt = f"""
You are an MS research assistant. Provide medically accurate answers using evidence.

Research Context:
{context}

Sources: {', '.join(sources)}

Question:
{query}

Answer:
"""

    resp = llm.invoke(prompt)
    return resp.content.strip()
