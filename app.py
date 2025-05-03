import streamlit as st
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Load secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX = "aped-4627"  # change if your index name is different
PINECONE_REGION = "us-west-2"  # update based on Pinecone dashboard
PINECONE_CLOUD = "aws"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='gcp',         
            region='gcp-starter' 
        )
    )


# Connect to the index
index = pc.Index(PINECONE_INDEX)

# Load and process PDF
def load_pdf(pdf_file):
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    return documents

# Generate embeddings
def generate_embeddings(texts):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return embeddings.embed_documents(texts)

# Upsert to Pinecone
def upsert_to_pinecone(texts, vectors):
    ids = [f"doc-{i}" for i in range(len(texts))]
    items = list(zip(ids, vectors))
    index.upsert(vectors=items)

# Query Pinecone
def query_pinecone(query_text):
    embed_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    query_vector = embed_model.embed_query(query_text)
    results = index.query(vector=query_vector, top_k=3, include_metadata=False)
    return results

# Streamlit UI
def main():
    st.title("📄 AI-Powered PDF Search Assistant")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        st.success("PDF uploaded successfully!")

        with st.spinner("Processing PDF..."):
            documents = load_pdf(uploaded_file)
            texts = [doc.page_content for doc in documents]
            vectors = generate_embeddings(texts)
            upsert_to_pinecone(texts, vectors)
            st.success("Embeddings uploaded to Pinecone!")

    query = st.text_input("Ask a question about the PDF:")
    if query:
        with st.spinner("Searching..."):
            result = query_pinecone(query)
            st.subheader("🔍 Top Results:")
            for match in result.matches:
                st.write(f"Score: {match.score} — ID: {match.id}")

if __name__ == "__main__":
    main()
