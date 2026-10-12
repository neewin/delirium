"""
Test script for Module 2: Embedding Generation and Semantic Retrieval.
Run with: python test_retriever.py
"""

from retriever import SemanticRetriever

INDEX_PATH = "data/faiss_index/index.faiss"
METADATA_PATH = "data/metadata/metadata.json"

sample_chunks = [
    {"chunk_id": 1, "source": "sample.pdf", "page": 1,
     "text": "Machine learning is a branch of artificial intelligence."},
    {"chunk_id": 2, "source": "sample.pdf", "page": 2,
     "text": "Supervised learning uses labelled training data."},
    {"chunk_id": 3, "source": "sample.pdf", "page": 3,
     "text": "Neural networks are computational models inspired by biological neurons."},
]


def print_results(results) -> None:
    for rank, r in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"  Chunk ID : {r['chunk_id']}")
        print(f"  Source   : {r['source']}")
        print(f"  Page     : {r['page']}")
        print(f"  Score    : {r['score']:.4f}")
        print(f"  Text     : {r['text']}")


def main() -> None:
    question = "What is supervised learning?"

    print("1. Building retriever and index from sample chunks...")
    retriever = SemanticRetriever()
    retriever.build_index(sample_chunks)
    print(f"   Index built with {retriever.index.ntotal} vectors.")

    print(f"\n2. Searching for: '{question}'")
    results = retriever.search(question, top_k=3)
    print_results(results)

    print("\n3. Saving index and metadata to disk...")
    retriever.save(INDEX_PATH, METADATA_PATH)
    print(f"   Saved to {INDEX_PATH} and {METADATA_PATH}")

    print("\n4. Loading index and metadata into a fresh retriever...")
    reloaded = SemanticRetriever()
    reloaded.load(INDEX_PATH, METADATA_PATH)
    print(f"   Loaded index with {reloaded.index.ntotal} vectors.")

    print(f"\n5. Searching again after reload: '{question}'")
    results_after_reload = reloaded.search(question, top_k=3)
    print_results(results_after_reload)


if __name__ == "__main__":
    main()
