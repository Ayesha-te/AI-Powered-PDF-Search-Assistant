import os
import openai
import numpy as np
import streamlit as st

from pinecone import Pinecone, ServerlessSpec
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
import faiss

# Load secrets from Streamlit
config = st.secrets

# Set API keys
openai.api_key = config["openai"]["api_key"]

# Pinecone setup
pc = Pinecone(api_key=config["pinecone"]["api_key"])
PINECONE_INDEX = config["pinecone"]["index_name"]
REGION = config["pinecone"]["region"]
CLOUD = config["pinecone"]["cloud"]

# Create index if it doesn't exist
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=CLOUD,
            region=REGION
        )
    )

pinecone_index = pc.Index(PINECONE_INDEX)

# Step 1: Load PDF and Extract Text
def load_pdf(uploaded_file):
    loader = PyPDFLoader(uploaded_file.name)
    documents = loader.load()
    return documents

# Step 2: Generate Embeddings
def generate_embeddings(documents):
    embedder = OpenAIEmbeddings()
    return [embedder.embed_query(doc.page_content) for doc in documents]

# Step 3: Store in FAISS
def store_embeddings_faiss(embeddings):
    vectors = np.array(embeddings).astype("float32")
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

# Step 4: Search FAISS
def search(query, index, embedder, top_k=5):
    query_vector = np.array(embedder.embed_query(query)).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    return indices[0]

# Step 5: Streamlit App
def main():
    st.title("📄 AI-Powered PDF Search Assistant")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        documents = load_pdf(uploaded_file)
        embeddings = generate_embeddings(documents)
        index = store_embeddings_faiss(embeddings)

        query = st.text_input("Ask something from the PDF:")
        if query:
            embedder = OpenAIEmbeddings()
            result_indices = search(query, index, embedder)
            st.subheader("🔍 Top Matching Pages:")
            for idx in result_indices:
                st.write(documents[idx].page_content[:500] + "...")


if __name__ == "__main__":
    main()
