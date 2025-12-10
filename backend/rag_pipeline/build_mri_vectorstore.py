# backend/rag_pipeline/build_mri_vectorstore.py

import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Use project-root-safe path instead of D:/ (IMPORTANT for deployment)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "..", "..", "vectorstores", "mri_db_free")


def build_mri_db():
    # Create an empty Chroma DB
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
    )

    vectordb.persist()
    print("✅ MRI vectorstore initialized (empty, no images added).")

if __name__ == "__main__":
    build_mri_db()
