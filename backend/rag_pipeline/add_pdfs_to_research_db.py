# backend/rag_pipeline/add_pdfs_to_research_db.py

import os
import hashlib

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ============================================================
# 📁 PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PAPERS_DIR = os.path.join("backend", "papers")  
PERSIST_DIR = os.path.join("backend", "vectorstores", "research_db_free")

os.makedirs(PERSIST_DIR, exist_ok=True)

embedding = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")


# ============================================================
# 🔐 HASHING FUNCTION
# ============================================================
def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ============================================================
# 📦 LOAD EXISTING HASHES
# ============================================================
def get_existing_hashes(vectordb):
    hashes = set()
    try:
        all_data = vectordb.get(include=["metadatas"])
        for m in all_data["metadatas"]:
            if m and "hash" in m:
                hashes.add(m["hash"])
    except Exception:
        pass
    return hashes


# ============================================================
# 🧠 MAIN FUNCTION — Adds all PDFs from papers/
# ============================================================
def add_pdfs_to_vectorstore():
    print("🚀 Updating research vectorstore with new PDFs...")

    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding
    )

    existing_hashes = get_existing_hashes(vectordb)
    print(f"🔍 Existing chunks in vectorstore: {len(existing_hashes)}")

    # --------------------------------------------------------
    # 🧾 Get all PDFs in backend/papers
    # --------------------------------------------------------
    pdf_files = [
        os.path.join(PAPERS_DIR, f)
        for f in os.listdir(PAPERS_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("⚠️ No PDF files found in:", PAPERS_DIR)
        return

    print(f"📄 Found {len(pdf_files)} PDFs")

    # --------------------------------------------------------
    # 📚 Load PDFs
    # --------------------------------------------------------
    loaded_docs = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            for d in docs:
                d.metadata["source"] = os.path.basename(pdf_path)

            loaded_docs.extend(docs)
            print(f"✅ Loaded {len(docs)} pages from:", os.path.basename(pdf_path))

        except Exception as e:
            print(f"❌ Failed to load {pdf_path}: {e}")

    # --------------------------------------------------------
    # ✂️ Split into chunks
    # --------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )

    chunks = splitter.split_documents(loaded_docs)
    print(f"🧩 Generated {len(chunks)} chunks")

    # --------------------------------------------------------
    # 🚫 Remove duplicates using text hash
    # --------------------------------------------------------
    new_chunks = []
    for chunk in chunks:
        h = compute_hash(chunk.page_content)
        if h not in existing_hashes:
            chunk.metadata["hash"] = h
            new_chunks.append(chunk)

    print(f"✨ {len(new_chunks)} new unique chunks detected")

    # --------------------------------------------------------
    # 💾 Store in vectorstore
    # --------------------------------------------------------
    if new_chunks:
        vectordb.add_documents(new_chunks)
        print(f"💾 Added {len(new_chunks)} new chunks into DB")
    else:
        print("✅ No new chunks — already up to date.")

    print("🎉 Research PDF vectorstore updated successfully!")


# ============================================================
# 🔥 RUN DIRECTLY
# ============================================================
if __name__ == "__main__":
    add_pdfs_to_vectorstore()
