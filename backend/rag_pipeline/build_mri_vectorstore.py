# backend/rag_pipeline/build_mri_vectorstore.py

import os
from langchain_chroma import Chroma

BASE_DIR = os.path.join("backend", "vectorstores", "mri_db_free")

def build_mri_db():
    os.makedirs(BASE_DIR, exist_ok=True)
    Chroma(persist_directory=BASE_DIR)
    print("✅ MRI vectorstore initialized at:", BASE_DIR)

if __name__ == "__main__":
    build_mri_db()
