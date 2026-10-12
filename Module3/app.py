"""
Streamlit demo for Modules 1-3.
Run with: streamlit run app.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from Module1.document_processor import process_document
from Module2.retriever import SemanticRetriever
from answer_generator import generate_answer, STATUS_SUCCESS, STATUS_INSUFFICIENT

UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")


def save_uploaded_file(uploaded_file) -> str:
    """Save the uploaded PDF into uploads/ and return its path."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def main() -> None:
    st.title("RAG-Based Document Chatbot")
    st.caption("Module 1: Processing | Module 2: Retrieval | Module 3: Answer Generation")

    # Keep the retriever between reruns so the index is built only once.
    st.session_state.setdefault("retriever", None)
    st.session_state.setdefault("stats", (0, 0))

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is not None:
        pdf_path = save_uploaded_file(uploaded_file)
        st.success(f"Uploaded file: {uploaded_file.name}")

        if st.button("Process Document"):
            try:
                chunks = process_document(pdf_path)
                with st.spinner("Building embeddings and FAISS index..."):
                    retriever = SemanticRetriever()
                    retriever.build_index(chunks)
                st.session_state.retriever = retriever
                st.session_state.stats = (len({c["page"] for c in chunks}), len(chunks))
                st.success("Document processed successfully.")
            except Exception as e:
                st.error(f"Could not process document: {e}")

    pages, num_chunks = st.session_state.stats
    if num_chunks:
        st.write(f"**Pages:** {pages}  |  **Chunks:** {num_chunks}")

    st.divider()
    question = st.text_input("Ask a question about the document")
    top_k = st.number_input("Passages to retrieve", value=3, min_value=1, max_value=10)

    if st.button("Get Answer"):
        if st.session_state.retriever is None:
            st.error("Please upload and process a document first.")
            return
        if not question.strip():
            st.error("Please enter a question.")
            return

        try:
            retrieved = st.session_state.retriever.search(question, top_k=int(top_k))
        except Exception as e:
            st.error(f"Retrieval failed: {e}")
            return

        with st.spinner("Generating answer..."):
            result = generate_answer(question, retrieved)

        st.subheader("Answer")
        if result["status"] in (STATUS_SUCCESS, STATUS_INSUFFICIENT):
            st.write(result["answer"])
            st.warning("Generated from retrieved passages but NOT yet verified (Module 4 pending).")
        else:
            st.error(f"{result['answer']}  (status: {result['status']})")

        if result["sources"]:
            st.subheader("Sources")
            for s in result["sources"]:
                st.write(f"- {s['source']}, page {s['page']} (chunk {s['chunk_id']}, score {s['score']:.2f})")

        with st.expander("Retrieved evidence passages"):
            for i, r in enumerate(retrieved, start=1):
                st.markdown(f"**{i}. Page {r['page']} — score {r['score']:.2f}**")
                st.write(r["text"])


if __name__ == "__main__":
    main()
