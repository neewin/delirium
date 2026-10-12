"""
Simple test script for Module 1: Document Processing.
Usage: python test_processor.py path/to/sample.pdf
"""

import sys
from document_processor import process_document


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python test_processor.py <path_to_pdf>")
        return

    pdf_path = sys.argv[1]

    try:
        chunks = process_document(pdf_path, chunk_size=1000, chunk_overlap=200)
    except Exception as e:
        print(f"Error processing document: {e}")
        return

    print(f"Total chunks generated: {len(chunks)}\n")

    print("First 3 chunks:")
    for chunk in chunks[:3]:
        print("-" * 50)
        print(f"Chunk ID : {chunk['chunk_id']}")
        print(f"Source   : {chunk['source']}")
        print(f"Page     : {chunk['page']}")
        print(f"Text     : {chunk['text'][:200]}...")


if __name__ == "__main__":
    main()
