# backend/rag_pipeline/build_research_vectorstore.py
import os, itertools, hashlib
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, ArxivLoader, PubMedLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ==============================
# 🔧 CONFIG
# ==============================
PERSIST_DIR = "vectorstores/research_db_free"
embedding = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

# ==============================
# 🧠 RESEARCH SOURCES
# ==============================
NEW_RESEARCH_URLS = [
    # 🏥 General & Clinical Information
    "https://www.nhs.uk/conditions/multiple-sclerosis/",
    "https://www.mayoclinic.org/diseases-conditions/multiple-sclerosis/symptoms-causes/syc-20350269",
    "https://www.cdc.gov/multiple-sclerosis/index.html",
    "https://www.nationalmssociety.org/What-is-MS/Types-of-MS",
    "https://en.wikipedia.org/wiki/Multiple_sclerosis",
    "https://www.who.int/news-room/fact-sheets/detail/multiple-sclerosis",
    "https://www.clevelandclinicmeded.com/medicalpubs/diseasemanagement/neurology/multiple-sclerosis/",
    # 🧬 Pathophysiology & Immunology
    "https://www.frontiersin.org/journals/immunology/sections/multiple-sclerosis-and-neuroimmunology",
    "https://www.nature.com/articles/s41582-023-00899-2",
    "https://www.thelancet.com/journals/laneur/article/PIIS1474-4422(21)00323-2/fulltext",
    "https://pubmed.ncbi.nlm.nih.gov/36594979/",
    "https://jamanetwork.com/journals/jamaneurology/fullarticle/2798393",
    # 🧲 MRI & Imaging Studies
    "https://radiopaedia.org/articles/multiple-sclerosis-3",
    "https://www.sciencedirect.com/topics/medicine-and-dentistry/multiple-sclerosis-mri",
    "https://www.frontiersin.org/articles/10.3389/fneur.2022.1005813/full",
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8968296/",
    # 💊 Treatment & Clinical Guidelines
    "https://www.fda.gov/consumers/consumer-updates/treatments-multiple-sclerosis-ms",
    "https://www.ema.europa.eu/en/human-regulatory/overview/multiple-sclerosis-treatments",
    "https://emedicine.medscape.com/article/1146199-overview",
    "https://www.ninds.nih.gov/health-information/disorders/multiple-sclerosis",
    "https://www.uptodate.com/contents/overview-of-the-treatment-of-multiple-sclerosis-in-adults",
    "https://www.nejm.org/doi/full/10.1056/NEJMra1401483",
    # 🧪 Research & Epidemiology
    "https://multiplesclerosisnewstoday.com/",
    "https://msif.org/about-us/what-we-do/research/",
    "https://www.msif.org/about-ms/what-is-ms/",
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9119274/",
    "https://www.thelancet.com/journals/laneur/article/PIIS1474-4422(19)30225-2/fulltext",
]

# ==============================
# ⚙️ HELPERS
# ==============================
def compute_hash(text: str) -> str:
    """Create a unique hash for each chunk's text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_existing_hashes(vectordb):
    """Extract hashes from existing Chroma vectorstore (if any)."""
    hashes = set()
    try:
        data = vectordb.get(include=["metadatas"])
        for meta in data["metadatas"]:
            if meta and "hash" in meta:
                hashes.add(meta["hash"])
    except Exception:
        pass
    return hashes


# ==============================
# 🌐 LIVE FETCH (PubMed + ArXiv + Scholar)
# ==============================
def fetch_live_research(max_docs_per_source=5):
    docs = []
    ms_terms = ["multiple sclerosis", "MS disease", "demyelination disorder"]
    ai_terms = ["machine learning", "deep learning", "AI"]
    modality_terms = ["MRI", "magnetic resonance imaging", "lesion detection", "brain scan"]
    task_terms = ["classification", "diagnosis", "prediction", "segmentation", "progression", "treatment", "symptoms"]

    query_combinations = [
        f"{a} {b} {c} {d}"
        for a, b, c, d in itertools.product(ms_terms, ai_terms, modality_terms, task_terms)
    ][:8]

    for query in query_combinations:
        try:
            pubmed_loader = PubMedLoader(query=query, load_max_docs=max_docs_per_source)
            res = pubmed_loader.load()
            docs.extend(res)
            print(f"✅ {len(res)} from PubMed for '{query}'")
        except Exception as e:
            print(f"⚠️ PubMed failed for '{query}': {e}")

        try:
            arxiv_loader = ArxivLoader(query=query, load_max_docs=max_docs_per_source)
            res = arxiv_loader.load()
            docs.extend(res)
            print(f"✅ {len(res)} from ArXiv for '{query}'")
        except Exception as e:
            print(f"⚠️ ArXiv failed for '{query}': {e}")

    try:
        from langchain_community.document_loaders import GoogleScholarLoader
        for query in query_combinations:
            scholar_loader = GoogleScholarLoader(query=query, num_results=max_docs_per_source)
            res = scholar_loader.load()
            docs.extend(res)
            print(f"🎓 Added {len(res)} from Google Scholar for '{query}'")
    except Exception as e:
        print(f"⚠️ Google Scholar skipped (missing scholarly/SerpAPI): {e}")

    print(f"\n📚 Total live research docs fetched: {len(docs)}\n")
    return docs


# ==============================
# 🧩 BUILD FUNCTION (Incremental)
# ==============================
def build_research_db():
    print("🚀 Building or updating research vectorstore...")
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
    existing_hashes = get_existing_hashes(vectordb)
    print(f"🔍 Existing chunks in DB: {len(existing_hashes)}")

    all_docs = []

    # Load from curated URLs
    for url in NEW_RESEARCH_URLS:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = url
            all_docs.extend(docs)
            print(f"✅ Loaded: {url}")
        except Exception as e:
            print(f"⚠️ Skipped {url}: {e}")

    # Fetch dynamic research
    live_docs = fetch_live_research()
    all_docs.extend(live_docs)
    print(f"📚 Total documents (static + live): {len(all_docs)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)
    print(f"🧩 Generated {len(chunks)} text chunks.")

    # Deduplicate by hash
    new_chunks = []
    for c in chunks:
        h = compute_hash(c.page_content)
        if h not in existing_hashes:
            c.metadata["hash"] = h
            new_chunks.append(c)

    print(f"✨ {len(new_chunks)} new unique chunks detected.")

    if new_chunks:
        vectordb.add_documents(new_chunks)
        print(f"💾 Added {len(new_chunks)} new chunks to '{PERSIST_DIR}'")
    else:
        print("✅ No new chunks to add — vectorstore already up to date.")

    print("✅ Research vectorstore ready!")


# ==============================
# 🚀 MAIN
# ==============================
if __name__ == "__main__":
    build_research_db()
