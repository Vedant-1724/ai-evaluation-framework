
from pathlib import Path

from src.document_loader import load_all_documents
from src.chunker import chunk_documents
from src.embeddings import embed_texts
from src.vector_store import add_chunks, get_collection_count


def main() -> None:
    documents_dir = Path("data/documents")

    print("=" * 60)
    print("AI-7 | RAG DOCUMENT INGESTION")
    print("=" * 60)

    # 1. Load source documents
    documents = load_all_documents(documents_dir)

    if not documents:
        raise RuntimeError(
            f"No supported documents found in {documents_dir.resolve()}"
        )

    print(f"\nDocuments loaded: {len(documents)}")

    # 2. Split documents into overlapping chunks
    chunks = chunk_documents(documents)
    print(f"Chunks created: {len(chunks)}")

    if not chunks:
        raise RuntimeError("Chunking produced no chunks.")

    # 3. Generate embeddings for the chunks
    texts = [chunk["content"] for chunk in chunks]
    embeddings = embed_texts(texts)
    print(f"Embeddings generated: {len(embeddings)}")

    # 4. Store chunks and embeddings in AI-7's ChromaDB
    add_chunks(chunks, embeddings)

    stored_count = get_collection_count()

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Documents processed: {len(documents)}")
    print(f"Chunks processed:    {len(chunks)}")
    print(f"Embeddings created:  {len(embeddings)}")
    print(f"Chunks in database:  {stored_count}")
    print("Database location:   data/chroma")
    print("\nIngestion completed.")


if __name__ == "__main__":
    main()
