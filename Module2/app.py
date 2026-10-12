"""
Streamlit interface for Module 1 (Document Processing) and
Module 2 (Embedding Generation and Semantic Retrieval).
Run with: streamlit run app.py
"""

import os
import sys

# Add the Project root folder to Python's search path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from Module1.document_processor import process_document
from retriever import SemanticRetriever

UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")


def save_uploaded_file(uploaded_file) -> str:
    """Save an uploaded Streamlit file object to the uploads/ directory."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def main() -> None:
    st.title("RAG-Based Document Chatbot")
    st.caption("Module 1: Document Processing  |  Module 2: Semantic Retrieval")

    # Keep the retriever and chunk count alive across reruns/interactions.
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "num_chunks" not in st.session_state:
        st.session_state.num_chunks = 0
    if "num_pages" not in st.session_state:
        st.session_state.num_pages = 0

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        pdf_path = save_uploaded_file(uploaded_file)
        st.success(f"Uploaded file: {uploaded_file.name}")

        if st.button("Process Document"):
            try:
                chunks = process_document(pdf_path, chunk_size=1000, chunk_overlap=200)
            except Exception as e:
                st.error(f"Could not process PDF: {e}")
                return

            st.session_state.num_pages = len({c["page"] for c in chunks})
            st.session_state.num_chunks = len(chunks)

            with st.spinner("Building embeddings and FAISS index..."):
                try:
                    retriever = SemanticRetriever()
                    retriever.build_index(chunks)
                    st.session_state.retriever = retriever
                except Exception as e:
                    st.error(f"Could not build index: {e}")
                    return

            st.success("Document processed successfully.")

    if st.session_state.num_chunks:
        st.write(f"**Pages:** {st.session_state.num_pages}")
        st.write(f"**Chunks:** {st.session_state.num_chunks}")

    st.divider()
    st.subheader("Ask a question")
    question = st.text_input("Enter your question")
    top_k = st.number_input("Number of results", value=3, min_value=1, max_value=10)

    if st.button("Search"):
        if st.session_state.retriever is None:
            st.error("Please upload and process a document first.")
        elif not question.strip():
            st.error("Please enter a question.")
        else:
            try:
                results = st.session_state.retriever.search(question, top_k=int(top_k))
            except Exception as e:
                st.error(f"Search failed: {e}")
                return

            st.subheader("Retrieved Evidence")
            st.caption("These are retrieved passages, not a generated answer (Module 3 will add that).")
            for rank, r in enumerate(results, start=1):
                with st.expander(f"{rank}. Page {r['page']} — Score: {r['score']:.2f}"):
                    st.write(f"**Source:** {r['source']}")
                    st.write(f"**Chunk ID:** {r['chunk_id']}")
                    st.write(r["text"])


if __name__ == "__main__":
    main()
