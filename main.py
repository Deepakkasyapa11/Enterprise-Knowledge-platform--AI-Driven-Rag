# main.py (Updated snippet)
def run_pipeline():
    # ... (keep your documents and metadatas as they were)

    pipeline = IngestionPipeline()
    vector_db = pipeline.create_vector_store(documents, metadatas)
    
    doc_objects = [Document(page_content=d, metadata=m) for d, m in zip(documents, metadatas)]
    
    # Initialize our custom retriever
    retriever = HybridRetriever(vector_db, doc_objects)

    query = "How do I secure my login?"
    results = retriever.invoke(query) 

    print(f"\nQuery: {query}")
    for res in results:
        print(f"Retrieved: {res.page_content} [{res.metadata.get('source')}]")

    # ... (keep the evaluation logic)