# backend/rag_pipeline/build_research_vectorstore.py
import os
import itertools
from supabase_client import supabase
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    ArxivLoader,
    PubMedLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from supabase_client import supabase

# ==============================
# ⚙️ ENV + CONFIG
# ==============================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PERSIST_DIR = "vectorstores/research_db_free"
EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

RESEARCH_URLS = [
    "https://www.nhs.uk/conditions/multiple-sclerosis/",
    "https://www.mayoclinic.org/diseases-conditions/multiple-sclerosis/symptoms-causes/syc-20350269",
    "https://en.wikipedia.org/wiki/Multiple_sclerosis",
    "https://www.nationalmssociety.org/What-is-MS/Types-of-MS",
    "https://www.cdc.gov/multiple-sclerosis/index.html",
]

# ==============================
# 🧠 LOG TO SUPABASE
# ==============================
def log_research_sources(docs, model: str = EMBED_MODEL):
    """Safely logs metadata of research documents to Supabase in batches."""
    if not docs:
        print("⚠️ No documents to log to Supabase.")
        return

    entries = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}

        title = meta.get("title") or meta.get("source") or "Untitled Document"
        authors = meta.get("authors") or "Unknown"
        year = meta.get("year") or 2025
        source_url = meta.get("source") or meta.get("url")
        pdf_path = meta.get("file_path") or meta.get("path")

        entries.append({
            "title": str(title)[:500],
            "authors": str(authors)[:500],
            "year": year if isinstance(year, int) else 2025,
            "source_url": source_url,
            "pdf_path": pdf_path,
            "embedding_model": model,
            "chunk_count": 1,
        })

    # Batch insert (Supabase has 100-row insert limit)
    try:
        batch_size = 100
        for i in range(0, len(entries), batch_size):
            chunk = entries[i:i + batch_size]
            supabase.table("research_sources").insert(chunk).execute()
        print(f"✅ Logged {len(entries)} research sources to Supabase.")
    except Exception as e:
        print(f"❌ Failed to log research sources: {e}")

# ==============================
# 🌐 FETCH LIVE RESEARCH
# ==============================
def fetch_live_research(max_docs_per_source=5):
    """
    Fetches latest Multiple Sclerosis research from PubMed, arXiv, and Google Scholar.
    """
    docs = []
    ms_terms = ["multiple sclerosis", "MS disease", "demyelination disorder"]
    ai_terms = ["machine learning", "deep learning", "artificial intelligence", "AI"]
    modality_terms = ["MRI", "magnetic resonance imaging", "lesion detection", "brain scan"]
    task_terms = ["classification", "diagnosis", "prediction", "segmentation", "progression", "treatment", "symptoms"]

    query_combinations = [
        f"{a} {b} {c} {d}"
        for a, b, c, d in itertools.product(ms_terms, ai_terms, modality_terms, task_terms)
    ][:10]

    loaders = [
        ("PubMed", PubMedLoader),
        ("ArXiv", ArxivLoader),
    ]

    for source_name, LoaderClass in loaders:
        try:
            for query in query_combinations:
                print(f"🔎 {source_name}: {query}")
                loader = LoaderClass(query=query, load_max_docs=max_docs_per_source)
                results = loader.load()
                print(f"✅ {source_name}: {len(results)} docs for '{query}'")
                docs.extend(results)
        except Exception as e:
            print(f"⚠️ {source_name} fetch failed: {e}")

    # Google Scholar (optional)
    try:
        from langchain_community.document_loaders import GoogleScholarLoader
        for query in query_combinations:
            print(f"🎓 Scholar: {query}")
            loader = GoogleScholarLoader(query=query, num_results=max_docs_per_source)
            results = loader.load()
            print(f"✅ Scholar: {len(results)} results for '{query}'")
            docs.extend(results)
    except Exception as e:
        print(f"⚠️ Scholar fetch failed (optional): {e}")

    print(f"\n📚 Total live documents fetched: {len(docs)}\n")
    return docs

# ==============================
# 🧩 BUILD FUNCTION
# ==============================
def build_research_db():
    """Builds and persists the research vectorstore."""
    all_docs = []
    print(f"🔍 Loading {len(RESEARCH_URLS)} curated research URLs...")

    for url in RESEARCH_URLS:
        try:
            loader = WebBaseLoader(url)
            page_docs = loader.load()
            print(f"✅ Loaded {url} ({len(page_docs)} pages)")
            all_docs.extend(page_docs)
        except Exception as e:
            print(f"⚠️ Skipped {url}: {e}")

    live_docs = fetch_live_research()
    all_docs.extend(live_docs)
    print(f"📚 Total documents (static + live): {len(all_docs)}")

    if not all_docs:
        print("❌ No documents loaded — aborting vectorstore creation.")
        return

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,  # longer for better coherence
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", "?", "!", " ", ""]
)
    chunks = splitter.split_documents(all_docs)
    print(f"🧩 Split into {len(chunks)} chunks.")

    try:
        vectordb = Chroma.from_documents(
            chunks,
            embedding=embedding,
            persist_directory=PERSIST_DIR
        )
        vectordb.persist()
        print(f"💾 Saved {len(chunks)} chunks to '{PERSIST_DIR}'")
    except Exception as e:
        print(f"❌ Error creating Chroma vectorstore: {e}")
        return

    log_research_sources(all_docs)
    print("✅ Research vectorstore built and logged successfully!")

# ==============================
# 🚀 MAIN
# ==============================
if __name__ == "__main__":
    build_research_db()
