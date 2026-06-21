from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# Example usage
if __name__ == "__main__":
    import os
    
    # Check if data directory exists and has files
    data_dir = "RAG/data"
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        docs = load_all_documents(data_dir)
        if len(docs) > 0:
            print(f"[INFO] Loaded {len(docs)} documents")
            store = FaissVectorStore("RAG/faiss_store")
            store.build_from_documents(docs)
    
    try:
        rag_search = RAGSearch()
        query = "What is attention mechanism?"
        summary = rag_search.search_and_summarize(query, top_k=3)
        print("Summary:", summary)
    except Exception as e:
        print(f"[ERROR] Failed to run RAG search: {e}")
        print("[INFO] Make sure 'data' directory exists with PDF/TXT files")
