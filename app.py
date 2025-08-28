import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# Load .env variables
load_dotenv()
#hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#openai_key = os.getenv("OPENAI_API_KEY")
hf_key = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Streamlit UI setup
st.set_page_config(page_title="📚 RAG Chatbot", layout="wide")
st.title("📚 Ask Your Documents (RAG Demo)")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    os.makedirs("docs", exist_ok=True)
    with open(f"docs/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and process PDF
    loader = PyPDFLoader(f"docs/{uploaded_file.name}")
    pages = loader.load()
    st.success(f"Loaded {len(pages)} pages from {uploaded_file.name}")

    # Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    # Embeddings (using Hugging Face)
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vectorstore
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Retrieval chain
    retriever = vectorstore.as_retriever()

    # LLM (OpenAI)
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.2,
        openai_api_key=openai_key
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # User query
    question = st.text_input("Ask a question about your document:")
    if question:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(question)
        st.subheader("💡 Answer")
        st.write(answer)

