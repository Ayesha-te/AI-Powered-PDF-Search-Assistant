import os
import faiss
import numpy as np
import streamlit as st
import pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings

# ✅ Load API Keys from secrets.toml
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_env = st.secrets["PINECONE_ENV"]

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

# Step 1: Load PDF and Extract Text
def load_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file.name)
    documents = loader.load()
    return documents

# Step 2: Generate Embeddings using OpenAI
def generate_embeddings(documents):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    texts = [doc.page_content for doc in documents]
    return embeddings.embed_documents(texts)

# Step 3: Store Embeddings in FAISS
def store_embeddings_faiss(embedding_vectors):
    embedding_vectors = np.array(embedding_vectors).astype("float32")
    dimension = embedding_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_vectors)
    return index

# Step 4: Search
def search(query, faiss_index, embed_model, top_k=5):
    query_embedding = np.array(embed_model.embed_query(query)).astype("float32").reshape(1, -1)
    distances, indices = faiss_index.search(query_embedding, top_k)
    return indices

# Streamlit UI
def main():
    st.title("📄 AI-Powered PDF Search Assistant")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    if uploaded_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        documents = load_pdf(open("temp.pdf", "rb"))
        embeddings = generate_embeddings(documents)
        index = store_embeddings_faiss(embeddings)
        embed_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

        query = st.text_input("Ask something about the PDF:")
        if query:
            result_indices = search(query, index, embed_model)
            st.write(f"Top Match Indices: {result_indices}")

if __name__ == "__main__":
    main()
