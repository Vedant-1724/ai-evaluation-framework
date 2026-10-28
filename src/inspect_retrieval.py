
import sys
import json
from pathlib import Path

from src.embeddings import embed_text
from src.vector_store import search


# Ensure Unicode output works in the Windows terminal.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def retrieve(question: str, top_k: int = 10) -> list[dict]:
    """Retrieve the most similar chunks for a question."""
    query_embedding = embed_text(question)
    results = search(query_embedding, top_k=top_k)

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {
            "chunk_id": chunk_id,
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
            "distance": distance,
            "content": content,
        }
        for chunk_id, content, metadata, distance in zip(
            ids, documents, metadatas, distances
        )
    ]


def main() -> None:
    benchmark_path = Path("data/benchmark.json")

    with benchmark_path.open("r", encoding="utf-8-sig") as file:
        benchmark = json.load(file)

    # Inspect only the Generative AI benchmark cases.
    for case in benchmark:
        if not case["id"].startswith("python-"):
            continue

        print("\n" + "=" * 90)
        print(f"ID: {case['id']}")
        print(f"Question: {case['question']}")
        print(f"Expected sources: {case['relevant_sources']}")
        print("-" * 90)

        chunks = retrieve(case["question"], top_k=10)

        for rank, chunk in enumerate(chunks, start=1):
            print(f"\nRank: {rank}")
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Source: {chunk['source']}")
            print(f"Distance: {chunk['distance']:.4f}")
            print(f"Content: {chunk['content'][:700]}")

    print("\nRetrieval inspection completed.")


if __name__ == "__main__":
    main()
