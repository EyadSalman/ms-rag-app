# backend/rag_pipeline/agents.py

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import google.generativeai as genai

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("ERROR: GOOGLE_API_KEY is missing from environment variables!")

genai.configure(api_key=GOOGLE_API_KEY)

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

#  Initialize LLM and embeddings
embedding = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)


# Load vectorstores
MRI_DB = Chroma(persist_directory="vectorstores/mri_db_free", embedding_function=embedding)
RESEARCH_DB = Chroma(persist_directory="vectorstores/research_db_free", embedding_function=embedding)

def mri_agent(query: str):
    """Retrieve MRI examples similar to a query."""
    results = MRI_DB.similarity_search(query, k=3)
    context = "\n".join([r.page_content for r in results]) or "No MRI examples found."
    prompt = f"You are an MRI interpretation assistant.\n\n{context}\n\nQuestion: {query}\nAnswer:"
    resp = llm.invoke(prompt)
    print("Gemini MRI response raw:", resp)
    return {
  "answer": resp.content,
  "agent_type": "research",
  "rag_used": len(results) > 0,
  "sources": [r.metadata.get("source") for r in results]
}


def research_agent(query: str):
    """Retrieve relevant medical papers and generate a natural, well-cited answer (no PDF links shown)."""
    results = RESEARCH_DB.similarity_search(query, k=3)
    context = "\n".join([r.page_content for r in results]) or "No relevant papers found."

    # ✅ Collect only non-PDF source identifiers for prompting
    sources = []
    seen = set()
    for r in results:
        src = r.metadata.get("url") or r.metadata.get("source") or r.metadata.get("title")
        if src and not str(src).lower().endswith(".pdf") and src not in seen:  # 🔹 ignore PDF links
            sources.append(src)
            seen.add(src)
    source_text = ", ".join(sources[:3]) if sources else "no specific papers"

    # 🧠 Build better prompt
    prompt = f"""
You are a medical research assistant specialized in Multiple Sclerosis (MS).

Use the following research context and sources to answer the user's question naturally.
If the context is insufficient, say so and avoid fabricating details.

Write the answer like a human expert, naturally referencing studies (e.g.,
"According to a 2023 Nature study...") or or expriment results conducted related to Multiple Sclerosis.

Do not include or link to PDF files directly — simply refer to them conceptually if needed.

At the end, list paper names or journal names only, 
without clickable links.

### Research Context:
{context}

### Available Sources:
{source_text}

### Question:
{query}

### Answer:
"""

    resp = llm.invoke(prompt)
    print("AI Assistant response:", resp)

    if not resp.content.strip():
        retry = llm.invoke(f"Answer briefly, citing sources naturally if possible: {query}")
        return retry.content or "No response generated."

    # ✅ Append simplified sources (names only, no links)
    if sources:
        source_list = "\n".join([f"- {src}" for src in sources])
        final_answer = f"{resp.content.strip()}\n\n**Sources:**\n{source_list}"
    else:
        final_answer = resp.content.strip()

    return final_answer
