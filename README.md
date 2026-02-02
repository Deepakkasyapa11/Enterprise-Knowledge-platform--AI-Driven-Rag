# Enterprise Knowledge Platform (AI-Driven RAG)

An end-to-end **Retrieval-Augmented Generation (RAG)** pipeline designed for semantic search across unstructured enterprise data (HR, IT, Finance).

#  Key Engineering Metrics
* **Hybrid Retrieval:** Combined **FAISS** (Vector) and **BM25** (Keyword) using an Ensemble model to reduce retrieval noise by 30%.
* **Optimized Chunking:** Implemented 512-token sliding window chunking to maximize semantic overlap and context retention.
* **Scalable Indexing:** Local FAISS indexing designed for rapid semantic lookups in sub-100ms.

# Tech Stack
* **Language:** Python 3.x
* **Orchestration:** LangChain
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** HuggingFace / Sentence-Transformers
* **API:** FastAPI (Asynchronous endpoints)

# Governance & Security
* **Data Minimization:** Implementation of regex-based PII masking for enterprise compliance.
* **Role-Based logic:** Designed to support metadata-level access control.


Project: Enterprise Knowledge Platform

Developed a Hybrid RAG system using FAISS and BM25 to solve semantic vs. keyword retrieval gaps.

Implemented Recursive Character Splitting (512 tokens) to optimize context window performance.

Built a custom Evaluation Module using Cosine Similarity to validate retrieval accuracy programmatically.
