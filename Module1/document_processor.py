"""
Module 1: Document Processing
------------------------------
Handles PDF validation, text extraction, cleaning, and page-aware
sentence/paragraph-friendly chunking with metadata for a RAG pipeline.
"""

import os
import re
from typing import List, Dict


# ---------------------------------------------------------------------------
# 1. PDF Validation
# ---------------------------------------------------------------------------
def validate_pdf(pdf_path: str) -> None:
    """Raise an error if the given path is not a usable PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a .pdf file.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError("Only .pdf files are supported.")

    if os.path.getsize(pdf_path) == 0:
        raise ValueError("The uploaded PDF file is empty.")


# ---------------------------------------------------------------------------
# 2. PDF Text Extraction
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """Extract text from every page of a PDF using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of dicts like {"page": 1, "text": "..."} for each page.

    Raises:
        RuntimeError: If the PDF cannot be opened or read (e.g. corrupted).
    """
    import fitz  # PyMuPDF

    pages_data: List[Dict] = []

    try:
        pdf_document = fitz.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Could not open PDF (it may be corrupted): {exc}")

    try:
        if pdf_document.page_count == 0:
            raise RuntimeError("The PDF contains no pages.")

        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            try:
                page_text = page.get_text()
            except Exception:
                # Skip pages that fail extraction instead of crashing.
                page_text = ""

            pages_data.append({
                "page": page_number + 1,   # human-friendly 1-based page numbers
                "text": page_text or ""    # empty string if nothing extracted
            })
    finally:
        pdf_document.close()

    return pages_data


# ---------------------------------------------------------------------------
# 3. Text Cleaning
# ---------------------------------------------------------------------------
def clean_text(raw_text: str) -> str:
    """Clean extracted text without destroying its meaning.

    - Collapses multiple spaces/tabs into one.
    - Joins broken lines within the same paragraph.
    - Preserves paragraph breaks (blank lines) since they help
      paragraph-aware chunking later.

    Args:
        raw_text: Text as extracted from the PDF.

    Returns:
        Cleaned text.
    """
    if not raw_text:
        return ""

    # Normalize Windows/Mac line endings to \n
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # Split into paragraphs using blank lines as the separator
    raw_paragraphs = re.split(r"\n\s*\n", text)

    cleaned_paragraphs = []
    for paragraph in raw_paragraphs:
        # Within a paragraph, treat single line breaks as spaces
        # (PDFs often wrap sentences across lines).
        single_line = paragraph.replace("\n", " ")
        # Collapse multiple spaces/tabs into a single space
        single_line = re.sub(r"[ \t]+", " ", single_line)
        single_line = single_line.strip()
        if single_line:
            cleaned_paragraphs.append(single_line)

    # Rejoin paragraphs with a double newline to preserve structure
    return "\n\n".join(cleaned_paragraphs)


# ---------------------------------------------------------------------------
# 4. Text Chunking (page-aware, sentence/paragraph-friendly)
# ---------------------------------------------------------------------------
def _split_into_sentences(paragraph: str) -> List[str]:
    """Split a paragraph into sentences using simple punctuation rules.

    This avoids extra dependencies (like nltk) while still being smarter
    than a pure character cut.
    """
    # Split after '.', '!', or '?' followed by whitespace and a capital
    # letter or end of string. Keeps the punctuation attached.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(]|$)", paragraph)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    cleaned_text: str,
    page_number: int,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict]:
    """Split cleaned text from a single page into overlapping chunks.

    Chunking is paragraph- and sentence-aware: sentences are packed into
    a chunk until adding the next sentence would exceed chunk_size. Only
    if a single sentence itself is longer than chunk_size do we fall back
    to a character-based split for that sentence.

    Args:
        cleaned_text: Cleaned text belonging to one PDF page.
        page_number: The page this text came from (kept for metadata).
        chunk_size: Target maximum number of characters per chunk.
        chunk_overlap: Number of trailing characters repeated at the
            start of the next chunk, to preserve context across chunks.

    Returns:
        A list of dicts: {"page": page_number, "text": chunk_text}
    """
    if not cleaned_text:
        return []

    # Gather all sentences across all paragraphs, in order.
    paragraphs = cleaned_text.split("\n\n")
    sentences: List[str] = []
    for paragraph in paragraphs:
        sentences.extend(_split_into_sentences(paragraph))

    if not sentences:
        return []

    chunks: List[str] = []
    current_chunk = ""

    for sentence in sentences:
        # Case 1: a single sentence is longer than chunk_size on its own.
        if len(sentence) > chunk_size:
            # Flush whatever is currently building up.
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Fall back to character-based split just for this sentence.
            start = 0
            while start < len(sentence):
                end = start + chunk_size
                chunks.append(sentence[start:end].strip())
                start = end - chunk_overlap if end - chunk_overlap > start else end
            continue

        # Case 2: adding this sentence would overflow the current chunk.
        candidate = (current_chunk + " " + sentence).strip() if current_chunk else sentence
        if len(candidate) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap taken from the end of the last chunk.
            overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_chunk = (overlap_text + " " + sentence).strip()
        else:
            current_chunk = candidate

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Remove any empty chunks and build final structure with page info.
    return [{"page": page_number, "text": c} for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# 5 & 6. Metadata Assignment + Main Processing Function
# ---------------------------------------------------------------------------
def process_document(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict]:
    """Run the full Module 1 pipeline on a PDF and return processed chunks.

    Pipeline: validate -> extract -> clean -> chunk -> attach metadata.

    Args:
        pdf_path: Path to the PDF file to process.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Overlap (in characters) between consecutive chunks.

    Returns:
        List of chunk dicts, each with: chunk_id, source, page, text.

    Raises:
        FileNotFoundError, ValueError, RuntimeError: on invalid/corrupt PDFs.
    """
    # Step 1: Validate the PDF file itself.
    validate_pdf(pdf_path)

    # Step 2: Extract raw text per page.
    pages_data = extract_text_from_pdf(pdf_path)

    source_filename = os.path.basename(pdf_path)
    all_chunks: List[Dict] = []
    chunk_id_counter = 1
    any_text_found = False

    # Step 3 & 4: Clean and chunk text, page by page (keeps page info correct).
    for page_data in pages_data:
        cleaned = clean_text(page_data["text"])
        if not cleaned:
            continue  # skip empty/unreadable pages, don't crash

        any_text_found = True
        page_chunks = chunk_text(
            cleaned,
            page_number=page_data["page"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Step 5: Attach metadata to every chunk.
        for chunk in page_chunks:
            all_chunks.append({
                "chunk_id": chunk_id_counter,
                "source": source_filename,
                "page": chunk["page"],
                "text": chunk["text"],
            })
            chunk_id_counter += 1

    if not any_text_found:
        raise RuntimeError(
            "No extractable text was found in this PDF. "
            "It may be a scanned/image-only document."
        )

    return all_chunks
