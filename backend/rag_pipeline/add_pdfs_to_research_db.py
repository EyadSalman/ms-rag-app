# backend/rag_pipeline/add_pdfs_to_research_db.py
import os, hashlib
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==========================
# 🔧 CONFIGURATION
# ==========================
PERSIST_DIR = "vectorstores/research_db_free"
embedding = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

# 🧩 Previously added PDFs (for reference only)
PDF_FILES = [
    "papers/Euro J of Neurology - 2018 - Dobson - Multiple sclerosis   a review.pdf",
    "papers/Exploration of machine learning techniques in.pdf",
    "papers/fimmu-12-700582.pdf",
    "papers/journals.plos disability progression.pdf",
    "papers/Machine_Learning_Approaches_in_Study_of_Multiple_S.pdf",
    "papers/mdpi prognostic.pdf",
    "papers/nature disease progression.pdf",
    "papers/royal-society-machine-learning-for-refining-interpretation-of-magnetic-resonance-imaging-scans.pdf",
    "papers/science systematic review.pdf",
    "papers/Machine learning in diagnosis and disability prediction of multiple sclerosis.pdf",
]

# 🆕 Newly added PDFs (only these will be processed)
NEW_PDF_FILES = [
    "papers/sensors-22-07856.pdf",
    "papers/integrating-large-language-models-in-care-research-and-education-in-multiple-sclerosis-management.pdf",
    "papers/New tool for diagnosis.pdf",
    "papers/An Automatic Segmentation of T2-FLAIR Multiple.pdf",
    "papers/Artificial intelligence to predict clinical disability in patients.pdf",
    "papers/Role of MRI in MS.pdf",
    "papers/MRI in the Diagnosis and Monitoring of Multiple Sclerosis.pdf",
    "papers/the-neuropsychiatry-of-multiple-sclerosis.pdf",
    "papers/ptj3703175.pdf",
    "papers/Detecting New Lesions Using a Large Language Model  Applications in Real‐World.pdf",
]


# ==========================
# ⚙️ HELPER FUNCTIONS
# ==========================
def compute_hash(text: str) -> str:
    """Create a unique hash for each chunk's text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def add_pdfs_to_vectorstore():
    # Load existing Chroma DB
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)

    # Gather existing hashes (if present)
    existing_hashes = set()
    try:
        all_docs = vectordb.get(include=["metadatas"])
        for m in all_docs["metadatas"]:
            if m and "hash" in m:
                existing_hashes.add(m["hash"])
    except Exception:
        pass

    docs = []
    for path in NEW_PDF_FILES:
        if not os.path.exists(path):
            print(f"⚠️ Missing file: {path}")
            continue

        loader = PyPDFLoader(path)
        loaded_docs = loader.load()
        for d in loaded_docs:
            d.metadata["source"] = os.path.basename(path)
        docs.extend(loaded_docs)
        print(f"✅ Loaded {len(loaded_docs)} pages from {os.path.basename(path)}")

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,  # longer for better coherence
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", "?", "!", " ", ""]
)
    chunks = splitter.split_documents(docs)
    print(f"🧩 Split {len(chunks)} chunks from NEW PDFs")

    # Deduplicate before adding
    new_chunks = []
    for c in chunks:
        h = compute_hash(c.page_content)
        if h not in existing_hashes:
            c.metadata["hash"] = h
            new_chunks.append(c)
        else:
            continue

    print(f"🔎 {len(new_chunks)} new unique chunks detected (out of {len(chunks)}).")

    if new_chunks:
        vectordb.add_documents(new_chunks)
        print(f"💾 Added {len(new_chunks)} chunks to '{PERSIST_DIR}'")
    else:
        print("✅ No new chunks to add — vectorstore already up to date.")

    print("✅ New PDFs merged into research vectorstore successfully!")


# ==========================
# 🚀 MAIN
# ==========================
if __name__ == "__main__":
    add_pdfs_to_vectorstore()
