# retrieval.py
from langchain_community.retrievers import BM25Retriever

class HybridRetriever:
    def __init__(self, vector_db, docs):
        self.bm25 = BM25Retriever.from_documents(docs)
        self.bm25.k = 2
        self.faiss = vector_db.as_retriever(search_kwargs={"k": 2})

    def invoke(self, query):
        """
        Manually implements Weighted Ensemble Retrieval.
        Combines Keyword (BM25) and Semantic (FAISS) results.
        """
        # Get results from both systems
        bm25_results = self.bm25.invoke(query)
        faiss_results = self.faiss.invoke(query)

        # Merge results and remove duplicates (preserving order)
        combined = []
        seen_content = set()

        for doc in faiss_results + bm25_results:
            if doc.page_content not in seen_content:
                combined.append(doc)
                seen_content.add(doc.page_content)
        
        return combined[:3] # Return top 3 unique results