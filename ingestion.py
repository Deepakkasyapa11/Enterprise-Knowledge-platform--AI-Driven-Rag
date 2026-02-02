from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class IngestionPipeline:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def create_vector_store(self, texts, metadata):
        chunks = self.splitter.create_documents(texts, metadatas=metadata)
        return FAISS.from_documents(chunks, self.embeddings)