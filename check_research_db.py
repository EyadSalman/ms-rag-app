# check_research_db.py

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def check_vectorstore():
    print("🔍 Checking Research Vectorstore...")

    # Same embedding model as your agents.py
    embedding = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

    # Load persisted database
    db = Chroma(
        persist_directory="vectorstores/research_db_free",
        embedding_function=embedding
    )

    try:
        count = db._collection.count()
        print(f"📦 Total documents in store: {count}")

        # Try a test query
        query = "What are the symptoms of multiple sclerosis?"
        results = db.similarity_search(query, k=2)
        print(f"🔎 Retrieved docs: {len(results)}")

        for i, r in enumerate(results, 1):
            print(f"\n--- Doc {i} ---")
            print(r.page_content[:500])  # print first 500 chars

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_vectorstore()
