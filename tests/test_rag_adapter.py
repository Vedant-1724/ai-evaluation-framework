
from unittest.mock import patch

import pytest

from src.adapters.rag_adapter import RAGAdapter


def test_default_top_k() -> None:
    adapter = RAGAdapter()
    assert adapter.default_top_k == 5


def test_custom_top_k() -> None:
    adapter = RAGAdapter(default_top_k=3)
    assert adapter.default_top_k == 3


def test_invalid_default_top_k() -> None:
    with pytest.raises(ValueError):
        RAGAdapter(default_top_k=0)


def test_retrieve_delegates_to_rag_retriever() -> None:
    expected = [
        {
            "chunk_id": "doc.md::chunk_0",
            "content": "RAG combines retrieval and generation.",
            "source": "doc.md",
            "chunk_index": 0,
            "distance": 0.15,
        }
    ]

    with patch(
        "src.adapters.rag_adapter.retrieve",
        return_value=expected,
    ) as mock_retrieve:
        adapter = RAGAdapter(default_top_k=3)
        results = adapter.retrieve("What is RAG?")

    assert results == expected
    mock_retrieve.assert_called_once_with(
        query="What is RAG?",
        top_k=3,
    )


def test_retrieve_rejects_empty_question() -> None:
    adapter = RAGAdapter()

    with pytest.raises(ValueError):
        adapter.retrieve("   ")


def test_retrieve_rejects_invalid_top_k() -> None:
    adapter = RAGAdapter()

    with pytest.raises(ValueError):
        adapter.retrieve("What is RAG?", top_k=0)
