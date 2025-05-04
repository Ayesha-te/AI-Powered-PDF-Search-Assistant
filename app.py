import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Load secrets from Streamlit
config = st.secrets

# Step 1: Load PDF and split into chunks
def load_pdf(file):
    loader = PyPDFLoader(file.name)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(pages)
    return documents

# Step 2: Create FAISS vector index
def create_vector_index(documents, openai_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    index = FAISS.from_documents(documents, embeddings)
    return index

# Step 3: Query the vector index
def query_index(index, query, openai_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    docs = index.similarity_search(query)
    llm = OpenAI(temperature=0, openai_api_key=openai_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain.run(input_documents=docs, question=query)

# Streamlit UI
def main():
    st.set_page_config(page_title="AI PDF Search Assistant", layout="centered")
    st.title("📄 AI-Powered PDF Search Assistant")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            documents = load_pdf(uploaded_file)
            index = create_vector_index(documents, config["openai"]["api_key"])
            st.success("PDF processed and indexed!")

            query = st.text_input("Ask a question about the document:")
            if query:
                with st.spinner("Searching for answer..."):
                    answer = query_index(index, query, config["openai"]["api_key"])
                    st.subheader("Answer:")
                    st.write(answer)

if __name__ == "__main__":
    main()
