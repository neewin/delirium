"""
Streamlit interface for testing Module 1: Document Processing.
Run with: streamlit run app.py
"""

import os
import streamlit as st
from document_processor import process_document

UPLOAD_DIR = "uploads"


def save_uploaded_file(uploaded_file) -> str:
    """Save an uploaded Streamlit file object to the uploads/ directory.

    Returns:
        The path where the file was saved.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def main() -> None:
    st.title("RAG Document Chatbot — Module 1: Document Processing")
    st.write("Upload a PDF to extract, clean, and chunk its text.")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is None:
        return

    if not uploaded_file.name.lower().endswith(".pdf"):
        st.error("Please upload a valid .pdf file.")
        return

    st.success(f"Uploaded file: {uploaded_file.name}")
    pdf_path = save_uploaded_file(uploaded_file)

    chunk_size = st.number_input("Chunk size (characters)", value=1000, min_value=100)
    chunk_overlap = st.number_input("Chunk overlap (characters)", value=200, min_value=0)

    if st.button("Process Document"):
        try:
            chunks = process_document(
                pdf_path,
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
            )
        except FileNotFoundError:
            st.error("The uploaded file could not be found. Please try again.")
            return
        except ValueError as e:
            st.error(f"Invalid file: {e}")
            return
        except RuntimeError as e:
            st.error(f"Could not process PDF: {e}")
            return
        except Exception:
            st.error("An unexpected error occurred while processing the PDF.")
            return

        pages_processed = len({c["page"] for c in chunks})
        st.write(f"**Pages processed:** {pages_processed}")
        st.write(f"**Total chunks generated:** {len(chunks)}")

        st.subheader("Sample Chunks")
        for chunk in chunks[:5]:
            with st.expander(f"Chunk {chunk['chunk_id']} (Page {chunk['page']})"):
                st.write(f"**Source:** {chunk['source']}")
                st.write(f"**Page:** {chunk['page']}")
                st.write(f"**Text:** {chunk['text']}")


if __name__ == "__main__":
    main()
