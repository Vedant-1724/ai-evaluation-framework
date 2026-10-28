
from typing import Any

from src.retriever import retrieve


class RAGAdapter:
    """Adapter that exposes the RAG retriever to the evaluation engine."""

    def __init__(self, default_top_k: int = 5) -> None:
        if default_top_k < 1:
            raise ValueError("default_top_k must be at least 1.")

        self.default_top_k = default_top_k

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve ranked document chunks for a benchmark question."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        k = self.default_top_k if top_k is None else top_k

        if k < 1:
            raise ValueError("top_k must be at least 1.")

        return retrieve(query=question, top_k=k)
