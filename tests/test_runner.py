
import json

import pytest

from src.evaluation.runner import evaluate_benchmark


class FakeAdapter:
    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        data = {
            "What is RAG?": [
                {
                    "chunk_id": "rag.md::chunk_0",
                    "content": "RAG retrieves relevant information.",
                    "source": "rag.md",
                    "chunk_index": 0,
                    "distance": 0.1,
                },
                {
                    "chunk_id": "other.md::chunk_0",
                    "content": "Unrelated content.",
                    "source": "other.md",
                    "chunk_index": 0,
                    "distance": 0.8,
                },
            ]
        }
        return data.get(question, [])[:top_k]


def write_benchmark(tmp_path, cases):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    return path


def test_unlabelled_cases_do_not_receive_relevance_metrics(tmp_path):
    path = write_benchmark(
        tmp_path,
        [
            {
                "id": "case-1",
                "question": "What is RAG?",
                "expected_answer": (
                    "RAG retrieves relevant external information "
                    "to provide context to a language model."
                ),
                "relevant_sources": ["rag.md"],
                "relevant_chunk_ids": [],
                "status": "needs_ground_truth",
            }
        ],
    )

    report = evaluate_benchmark(path, adapter=FakeAdapter(), top_k=2)

    assert report["total_cases"] == 1
    assert report["labelled_cases"] == 0
    assert report["unlabelled_cases"] == 1
    assert report["results"][0]["metrics"] is None
    assert report["results"][0]["source_hit"] is True


def test_labelled_case_calculates_metrics(tmp_path):
    path = write_benchmark(
        tmp_path,
        [
            {
                "id": "case-1",
                "question": "What is RAG?",
                "expected_answer": (
                    "RAG retrieves relevant external information "
                    "to provide context to a language model."
                ),
                "relevant_sources": ["rag.md"],
                "relevant_chunk_ids": ["rag.md::chunk_0"],
                "status": "labelled",
            }
        ],
    )

    report = evaluate_benchmark(path, adapter=FakeAdapter(), top_k=2)

    metrics = report["results"][0]["metrics"]
    assert metrics["precision@2"] == 0.5
    assert metrics["recall@2"] == 1.0
    assert metrics["mrr"] == 1.0


def test_invalid_top_k_is_rejected(tmp_path):
    path = write_benchmark(
        tmp_path,
        [
            {
                "id": "case-1",
                "question": "What is RAG?",
                "expected_answer": "RAG retrieves relevant information.",
                "relevant_sources": [],
                "relevant_chunk_ids": [],
                "status": "needs_ground_truth",
            }
        ],
    )

    with pytest.raises(ValueError):
        evaluate_benchmark(path, adapter=FakeAdapter(), top_k=0)
