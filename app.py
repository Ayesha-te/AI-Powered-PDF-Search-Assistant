import os
import faiss
import openai
import pinecone
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
import toml

# Load API Keys from apikey.toml
config = toml.load("apikey.toml")
openai.api_key = config["openai"]["api_key"]
pinecone.init(api_key=config["pinecone"]["api_key"], environment=config["pinecone"]["environment"])

# Step 1: Load PDF and Extract Text
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

# Step 2: Generate Embeddings using OpenAI API
def generate_embeddings(documents):
    embeddings = OpenAIEmbeddings()
    document_embeddings = [embeddings.embed_document(doc) for doc in documents]
    return document_embeddings

# Step 3: Store Embeddings in FAISS
def store_embeddings_faiss(embeddings):
    embedding_vectors = np.array(embeddings)
    dimension = embedding_vectors.shape[1]  # Embedding size
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_vectors)
    return index

# Step 4: Store Embeddings in Pinecone (Optional, can be switched from FAISS)
def store_embeddings_pinecone(embeddings, index_name="pdf-search-index"):
    index = pinecone.Index(index_name)
    index.upsert(vectors=embeddings)
    return index

# Step 5: Search Function (Querying the embeddings)
def search(query, index, top_k=5):
    query_embedding = generate_embeddings([query])  # Generate query embedding
    distances, indices = index.search(query_embedding, top_k)
    return indices  # Return top-k most relevant documents

# Step 6: Streamlit Interface
def main():
    st.title("AI-Powered PDF Search Assistant")

    # File Upload Section
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if uploaded_file is not None:
        # Load PDF and generate embeddings
        documents = load_pdf(uploaded_file)
        embeddings = generate_embeddings(documents)
        
        # Store embeddings in FAISS
        index = store_embeddings_faiss(embeddings)

        # User Input: Ask a question
        query = st.text_input("Ask a question about the PDF:")
        if query:
            result_indices = search(query, index)
            st.write(f"Top Results: {result_indices}")

if __name__ == "__main__":
    main()
