# RAG-Based Document Chatbot — Module 1: Document Processing

This module handles PDF upload, text extraction, cleaning, and
page-aware, sentence/paragraph-friendly chunking with metadata.
Later modules (embeddings, FAISS, LLM generation, hallucination
detection) will consume the list of chunks returned here.

## Files

- `app.py` — Streamlit UI to upload a PDF and view generated chunks.
- `modules/document_processor.py` — Core logic (validate, extract, clean, chunk).
- `test_processor.py` — Command-line test script.
- `uploads/` — Stores uploaded PDFs.
- `requirements.txt` — Dependencies for Module 1 only.

## Install

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## Run the test script

```bash
python test_processor.py path/to/sample.pdf
```
