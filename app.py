import os
import numpy as np
import faiss
import openai
import toml
import streamlit as st

from pinecone import Pinecone, ServerlessSpec
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings

# Load API keys from apikey.toml
config = toml.load("apikey.toml")

openai.api_key = config["openai"]["api_key"]
pinecone_api_key = config["pinecone"]["api_key"]
pinecone_region = config["pinecone"]["region"] 
pinecone_cloud = config["pinecone"]["cloud"]    
pinecone_index_name = config["pinecone"]["index_name"]  

# Step 1: Load PDF and Extract Text
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return [doc.page_content for doc in documents]

# Step 2: Generate Embeddings
def generate_embeddings(text_chunks):
    embeddings = OpenAIEmbeddings()
    return embeddings.embed_documents(text_chunks)

# Step 3A: Store Embeddings in FAISS
def store_embeddings_faiss(embedding_vectors):
    vectors = np.array(embedding_vectors).astype("float32")
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

# Step 3B: Store Embeddings in Pinecone
def store_embeddings_pinecone(text_chunks, embedding_vectors):
    pc = Pinecone(api_key=pinecone_api_key)

    if pinecone_index_name not in pc.list_indexes().names():
        pc.create_index(
            name=pinecone_index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=pinecone_cloud,
                region=pinecone_region
            )
        )

    index = pc.Index(pinecone_index_name)

    vectors = [
        {
            "id": f"doc-{i}",
            "values": vector,
            "metadata": {"text": text_chunks[i]}
        }
        for i, vector in enumerate(embedding_vectors)
    ]

    index.upsert(vectors)
    return index

# Step 4: Search in FAISS
def search_faiss(query, text_chunks, faiss_index, top_k=5):
    query_embedding = generate_embeddings([query])[0]
    query_vector = np.array([query_embedding]).astype("float32")
    distances, indices = faiss_index.search(query_vector, top_k)
    results = [text_chunks[i] for i in indices[0]]
    return results

# Streamlit UI
def main():
    st.title("🔍 AI-Powered PDF Search Assistant")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        with st.spinner("Reading PDF and generating embeddings..."):
            documents = load_pdf(uploaded_file)
            embeddings = generate_embeddings(documents)

            try:
                # Try using Pinecone
                pinecone_index = store_embeddings_pinecone(documents, embeddings)
                use_pinecone = True
            except Exception as e:
                st.warning("⚠️ Pinecone error, using FAISS instead.")
                faiss_index = store_embeddings_faiss(embeddings)
                use_pinecone = False

        query = st.text_input("Ask something about the PDF:")
        if query:
            with st.spinner("Searching..."):
                if use_pinecone:
                    query_embedding = generate_embeddings([query])[0]
                    query_result = pinecone_index.query(
                        vector=query_embedding,
                        top_k=5,
                        include_metadata=True
                    )
                    results = [match["metadata"]["text"] for match in query_result["matches"]]
                else:
                    results = search_faiss(query, documents, faiss_index)

                st.subheader("Top Results:")
                for i, res in enumerate(results, 1):
                    st.markdown(f"**{i}.** {res}")

if __name__ == "__main__":
    main()
