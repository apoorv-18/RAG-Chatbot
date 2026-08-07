import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_core.language_models.llms import LLM
from typing import Optional, List
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
import tempfile
import os

load_dotenv()

class FlanT5LLM(LLM):
    """Minimal LangChain LLM wrapper around a local flan-t5 model, avoiding
    transformers' pipeline() factory which keeps changing task names across versions."""
    tokenizer: object
    model: object
    max_new_tokens: int = 300

    @property
    def _llm_type(self) -> str:
        return "flan-t5-local"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

st.set_page_config(page_title="RAG Document Q&A", layout="centered", page_icon="📄")
st.title("📄 RAG Document Q&A Chatbot")
st.caption("Upload a document and ask questions — powered by local embeddings + LLM")

# --- Sidebar: file upload ---
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Choose a .txt or .pdf file", type=["txt", "pdf"])
    st.markdown("---")
    st.caption("If no file is uploaded, the default sample.txt will be used.")

# --- Chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource(show_spinner="Setting up the knowledge base...")
def setup_qa_chain(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    if not chunks:
        st.error("⚠️ Could not extract any readable text from this file. It may be a scanned/image-based PDF. Please try a text-based PDF or a .txt file.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    llm = FlanT5LLM(tokenizer=tokenizer, model=model, max_new_tokens=300)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )
    return qa_chain

# --- Determine file to use ---
if uploaded_file is not None:
    suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        file_path = tmp.name
    file_type = "pdf" if suffix == ".pdf" else "txt"
    st.sidebar.success(f"Loaded: {uploaded_file.name}")
else:
    file_path = "sample.txt"
    file_type = "txt"

qa_chain = setup_qa_chain(file_path, file_type)

# --- Chat interface ---
query = st.chat_input("Ask a question about the document...")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"query": query})
        st.session_state.chat_history.append({
            "question": query,
            "answer": result["result"],
            "sources": [doc.page_content for doc in result["source_documents"]]
        })

# --- Display chat history (latest first) ---
for entry in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        with st.expander("📚 Source chunks used"):
            for src in entry["sources"]:
                st.markdown(f"> {src}")

if not st.session_state.chat_history:
    st.info("👋 Ask a question below to get started!")