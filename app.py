import tempfile

import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader


def get_openai_api_key() -> str:
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    if "openai_api_key" in st.secrets:
        return st.secrets["openai_api_key"]
    if "openai" in st.secrets:
        openai_section = st.secrets["openai"]
        if "apikey" in openai_section:
            return openai_section["apikey"]
        if "api_key" in openai_section:
            return openai_section["api_key"]

    st.error(
        "Missing OpenAI API key in Streamlit secrets. Add one of: "
        "`OPENAI_API_KEY`, `openai_api_key`, or `[openai] api_key`."
    )
    st.stop()


def get_client() -> OpenAI:
    return OpenAI(api_key=get_openai_api_key())


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [chunk for chunk in chunks if chunk.strip()]


def load_pdf(uploaded_file) -> list[str]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    reader = PdfReader(tmp_file_path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return chunk_text(full_text)


def embed_chunks(chunks: list[str]) -> np.ndarray:
    response = get_client().embeddings.create(model="text-embedding-3-small", input=chunks)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


def create_vector_index(chunks: list[str]):
    vectors = embed_chunks(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def query_index(index, chunks: list[str], query: str) -> str:
    query_embedding = get_client().embeddings.create(
        model="text-embedding-3-small",
        input=[query],
    )
    query_vector = np.array([query_embedding.data[0].embedding], dtype="float32")
    _, indices = index.search(query_vector, k=min(4, len(chunks)))
    relevant_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(relevant_chunks)

    response = get_client().chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Answer questions using only the supplied document context. Say when the document does not contain the answer.",
            },
            {
                "role": "user",
                "content": f"Document context:\n{context}\n\nQuestion: {query}",
            },
        ],
    )
    return response.choices[0].message.content or ""


def main():
    st.set_page_config(page_title="AI PDF Search Assistant", layout="centered")
    st.title("📄 AI-Powered PDF Search Assistant")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            chunks = load_pdf(uploaded_file)
            if not chunks:
                st.error("No extractable text was found in that PDF.")
                st.stop()

            index = create_vector_index(chunks)
            st.success("PDF processed and indexed!")

        query = st.text_input("Ask a question about the document:")
        if query:
            with st.spinner("Searching for answer..."):
                answer = query_index(index, chunks, query)
                st.subheader("Answer:")
                st.write(answer)


if __name__ == "__main__":
    main()
