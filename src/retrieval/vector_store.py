import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration Paths
PDF_DIR = "data/raw/pdf"
FAISS_INDEX_PATH = "data/processed/faiss_index"

def build_vector_store():
    print(f"Loading PDFs from {PDF_DIR}...")
    # 1. Load the unstructured PDF documents
    loader = PyPDFDirectoryLoader(PDF_DIR)
    documents = loader.load()
    
    if not documents:
        print("Error: No PDF documents found. Please run generate_pdf.py first.")
        return

    print(f"Loaded {len(documents)} document pages.")

    # 2. Split the documents into manageable semantic chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    # 3. Initialize the Embedding Model (Updated to langchain_huggingface)
    print("Downloading/Loading the embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Convert chunks to vectors and store in FAISS
    print("Generating embeddings and building the FAISS vector database...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 5. Save the FAISS index locally
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"Successfully saved FAISS index to {FAISS_INDEX_PATH}")

def test_retrieval(query):
    """Utility function to test if the vector store retrieves relevant chunks."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    print(f"\n--- Testing Retrieval for: '{query}' ---")
    results = vector_store.similarity_search(query, k=2)
    for i, doc in enumerate(results):
        print(f"\nResult {i+1} (Source: {doc.metadata['source']}):\n{doc.page_content}")

if __name__ == "__main__":
    # Build the database
    build_vector_store()
    
    # Test the database with a sample query
    test_retrieval("How many privilege leaves do I get in Bengaluru?")