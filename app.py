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

# Create index if it doesn't exist
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,
        metric="cosine",
