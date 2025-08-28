import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import HuggingFaceChat
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

st.set_page_config(page_title="📚 RAG Chatbot - HuggingFace", layout="wide")
st.title("📚 Ask Your PDF (HuggingFace)")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    st.success(f"Loaded {len(pages)} pages from {uploaded_file.name}")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.s_
