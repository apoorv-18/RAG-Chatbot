# 📄 RAG Document Q&A Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about uploaded documents (PDF or TXT) using local, open-source AI models — no paid API keys required.

## Overview

This project demonstrates a complete RAG pipeline: users upload a document, the system breaks it into chunks, converts those chunks into vector embeddings, stores them in a vector database, and retrieves the most relevant chunks to answer natural-language questions using a language model.

## Features

- 📁 Upload any .pdf or .txt document through the sidebar
- 💬 Chat-style interface with conversation history
- 🔍 Displays the exact source chunks used to generate each answer
- 🖥️ Runs entirely locally — no internet dependency for inference once models are downloaded
- 🆓 Built entirely with free, open-source models

## Tech Stack

- Frontend: Streamlit
- Orchestration: LangChain
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Vector Database: ChromaDB
- Language Model: google/flan-t5-base (local inference)
- Document Parsing: PyPDF

## Setup

`bash
python -m venv venv
venv\Scripts\activate
pip install streamlit langchain langchain-community langchain-huggingface sentence-transformers chromadb pypdf python-dotenv transformers torch
streamlit run app.py