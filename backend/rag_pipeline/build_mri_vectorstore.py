# backend/rag_pipeline/build_mri_vectorstore.py
import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

MRI_DIR = "D:/mss/MS_Merged_Datasets/test"
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def build_mri_db():
    docs = []
    for cls in ["MS", "Healthy"]:
        folder = os.path.join(MRI_DIR, cls)
        for f in os.listdir(folder):
            if f.endswith((".png", ".jpg", ".jpeg")):
                docs.append(Document(
                    page_content=f"MRI scan labeled {cls}",
                    metadata={"path": os.path.join(folder, f), "label": cls}
                ))

    vectordb = Chroma.from_documents(
        docs,
        embedding,
        persist_directory="vectorstores/mri_db_free"
    )
    vectordb.persist()
    print(f"✅ Saved MRI vector DB with {len(docs)} images.")
