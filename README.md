# 📄 RAG Document Q&A Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about uploaded documents (PDF or TXT) using local, open-source AI models — no paid API keys required.

## Overview

This project demonstrates a complete RAG pipeline: users upload a document, the system breaks it into chunks, converts those chunks into vector embeddings, stores them in a vector database, and retrieves the most relevant chunks to answer natural-language questions using a language model.

## Features

- 📁 Upload any `.pdf` or `.txt` document through the sidebar
- 💬 Chat-style interface with conversation history
- 🔍 Displays the exact source chunks used to generate each answer
- 🖥️ Runs entirely locally — no internet dependency for inference once models are downloaded
- 🆓 Built entirely with free, open-source models

## Tech Stack

| Component | Tool |
|---|---|
| Frontend | Streamlit |
| Orchestration | LangChain |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database | ChromaDB |
| Language Model | `google/flan-t5-small` (local inference) |
| Document Parsing | PyPDF |

## Requirements

- **Python 3.11** (recommended). Newer Python versions, such as 3.13+, may not yet have compatible `torch` wheels — if `pip install` fails to find a `torch` version, either install Python 3.11 or remove version pins from `requirements.txt` to let pip resolve compatible versions for your Python.

## Setup

```bash
# 1. Create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`. On first run, the embedding model and language model are downloaded from Hugging Face (internet required once); after that, everything runs offline.

## Usage

1. Upload a `.pdf` or `.txt` file from the sidebar (or use the included `sample.txt` to try it out immediately).
2. Type a question about the document in the chat box.
3. The app retrieves the most relevant chunks and generates an answer, showing the exact source text used underneath each response.

## Project Structure

```
gen-ai-rag-chatbot-main/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── runtime.txt          # Pinned Python version
├── sample.txt           # Sample document for quick testing
└── README.md
```

## How It Works

1. **Load** — the uploaded document is parsed with `PyPDFLoader` or `TextLoader`.
2. **Chunk** — text is split into overlapping chunks using `RecursiveCharacterTextSplitter`.
3. **Embed** — each chunk is converted into a vector using a sentence-transformers embedding model.
4. **Store** — vectors are stored in a local ChromaDB instance.
5. **Retrieve** — on each question, the most semantically similar chunks are retrieved.
6. **Generate** — the retrieved chunks and the question are passed to a local flan-t5 model to produce an answer.

## Improving Answer Quality

If answers seem inaccurate or generic:

- Swap `google/flan-t5-small` for `google/flan-t5-base` or `google/flan-t5-large` in `app.py` for stronger generation quality.
- Increase `search_kwargs={"k": 2}` to retrieve more chunks per question.
- Increase `chunk_size` and `chunk_overlap` in the text splitter so relevant context isn't cut off.
- Check the "📚 Source chunks used" expander under each answer to see exactly what context the model was given.

## Troubleshooting

- **`ModuleNotFoundError` for any LangChain/Transformers submodule** — these libraries have restructured their packages across recent versions (e.g. `RetrievalQA` now lives in `langchain_classic`, text splitters in `langchain_text_splitters`, Chroma in `langchain_chroma`). If you hit an import error, check the library's current docs for the module's new location, or pin `requirements.txt` to the versions this project was built against.
- **`streamlit` still using the wrong Python/packages** — confirm your virtual environment is activated (`(venv)` shown in your terminal prompt) before running `pip install` or `streamlit run`.
- **Scanned/image-based PDFs** — this project extracts text directly and does not perform OCR, so scanned PDFs with no embedded text layer won't work.

## License

No license specified.