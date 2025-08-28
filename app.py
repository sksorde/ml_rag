import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# Use secrets from Streamlit Cloud
openai_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="📚 RAG Chatbot", layout="wide")
st.title("📚 Ask Your PDF")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with open(f"temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and process PDF
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    st.success(f"Loaded {len(pages)} pages from {uploaded_file.name}")

    # Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    # Embeddings & Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # RAG chain
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key=openai_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Ask user input
    question = st.text_input("Ask a question about your document:")
    if question:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(question)
        st.subheader("💡 Answer")
        st.write(answer)
