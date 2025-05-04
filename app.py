# Step 2: Generate Embeddings
def generate_embeddings(documents, api_key):
    embedder = OpenAIEmbeddings(openai_api_key=api_key)
    return [embedder.embed_query(doc.page_content) for doc in documents]

# Step 5: Streamlit App
def main():
    st.title("📄 AI-Powered PDF Search Assistant")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        documents = load_pdf(uploaded_file)
        embeddings = generate_embeddings(documents, config["openai"]["api_key"])
        index = store_embeddings_faiss(embeddings)

        query = st.text_input("Ask something from the PDF:")
        if query:
            embedder = OpenAIEmbeddings(openai_api_key=config["openai"]["api_key"])
            result_indices = search(query, index, embedder)
            st.subheader("🔍 Top Matching Pages:")
            for idx in result_indices:
                st.write(documents[idx].page_content[:500] + "...")
