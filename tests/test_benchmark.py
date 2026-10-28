
import json

import pytest

from src.evaluation.benchmark import load_benchmark


def valid_case(**overrides):
    case = {
        "id": "case_001",
        "question": "What is RAG?",
        "relevant_sources": ["rag_reference.md"],
        "relevant_chunk_ids": ["rag_reference.md::chunk_0"],
        "expected_answer": (
            "RAG retrieves relevant external information and provides "
            "it as context to a language model."
        ),
        "status": "labelled",
    }
    case.update(overrides)
    return case


def test_load_valid_benchmark(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text(
        json.dumps([valid_case()]),
        encoding="utf-8",
    )

    result = load_benchmark(path)

    assert len(result) == 1
    assert result[0]["id"] == "case_001"
    assert result[0]["expected_answer"]


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_benchmark("missing_benchmark.json")


def test_invalid_json(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_benchmark(path)


def test_benchmark_must_be_list(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text('{"id": "case_001"}', encoding="utf-8")

    with pytest.raises(ValueError, match="JSON list"):
        load_benchmark(path)


def test_empty_benchmark_rejected(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="at least one case"):
        load_benchmark(path)


def test_duplicate_ids_rejected(tmp_path):
    case = valid_case()

    path = tmp_path / "benchmark.json"
    path.write_text(
        json.dumps([case, case]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate benchmark id"):
        load_benchmark(path)


def test_missing_question_rejected(tmp_path):
    case = valid_case()
    del case["question"]

    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps([case]), encoding="utf-8")

    with pytest.raises(ValueError, match="question"):
        load_benchmark(path)


def test_missing_expected_answer_rejected(tmp_path):
    case = valid_case()
    del case["expected_answer"]

    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps([case]), encoding="utf-8")

    with pytest.raises(ValueError, match="expected_answer"):
        load_benchmark(path)


def test_empty_expected_answer_rejected(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text(
        json.dumps([valid_case(expected_answer="  ")]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected_answer"):
        load_benchmark(path)


def test_labelled_case_requires_relevant_chunks(tmp_path):
    case = valid_case(relevant_chunk_ids=[])

    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps([case]), encoding="utf-8")

    with pytest.raises(ValueError, match="at least one relevant chunk"):
        load_benchmark(path)


def test_needs_ground_truth_can_have_no_relevant_chunks(tmp_path):
    case = valid_case(
        relevant_chunk_ids=[],
        status="needs_ground_truth",
    )

    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps([case]), encoding="utf-8")

    result = load_benchmark(path)

    assert result[0]["status"] == "needs_ground_truth"
