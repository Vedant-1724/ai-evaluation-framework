
import json
from pathlib import Path
from typing import Any


def load_benchmark(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a retrieval evaluation benchmark."""
    benchmark_path = Path(path)

    if not benchmark_path.is_file():
        raise FileNotFoundError(
            f"Benchmark file not found: {benchmark_path}"
        )

    with benchmark_path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Benchmark must be a JSON list")

    if not data:
        raise ValueError("Benchmark must contain at least one case")

    seen_ids = set()

    for index, case in enumerate(data):
        if not isinstance(case, dict):
            raise ValueError(f"Case {index} must be a JSON object")

        case_id = case.get("id")

        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(
                f"Case {index} must have a non-empty string id"
            )

        if case_id in seen_ids:
            raise ValueError(f"Duplicate benchmark id: {case_id}")

        seen_ids.add(case_id)

        question = case.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Case {case_id} must have a non-empty question"
            )

        for field in ("relevant_sources", "relevant_chunk_ids"):
            value = case.get(field)

            if not isinstance(value, list):
                raise ValueError(
                    f"Case {case_id}: {field} must be a list"
                )

            if not all(
                isinstance(item, str) and item.strip()
                for item in value
            ):
                raise ValueError(
                    f"Case {case_id}: {field} must contain "
                    "non-empty strings"
                )

        expected_answer = case.get("expected_answer")
        if not isinstance(expected_answer, str) or not expected_answer.strip():
            raise ValueError(
                f"Case {case_id} must have a non-empty expected_answer"
            )

        status = case.get("status")
        if status not in ("labelled", "needs_ground_truth"):
            raise ValueError(
                f"Case {case_id}: status must be "
                "'labelled' or 'needs_ground_truth'"
            )

        if status == "labelled" and not case["relevant_chunk_ids"]:
            raise ValueError(
                f"Case {case_id}: labelled cases must have "
                "at least one relevant chunk ID"
            )

    return data
