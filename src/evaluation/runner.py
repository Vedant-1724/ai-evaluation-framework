
"""Run retrieval evaluation against a labelled benchmark.

Usage:
    python -m src.evaluation.runner
    python -m src.evaluation.runner --top-k 10
    python -m src.evaluation.runner --benchmark data/benchmark.json
    python -m src.evaluation.runner --output reports/evaluation_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.adapters.rag_adapter import RAGAdapter
from src.evaluation.benchmark import load_benchmark
from src.evaluation.metrics import (
    duplicate_rate,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = PROJECT_ROOT / "data" / "benchmark.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "evaluation_report.json"


def _validate_retrieved_chunks(chunks: Any, case_id: str) -> list[dict[str, Any]]:
    """Validate the structure returned by the retrieval adapter."""
    if not isinstance(chunks, list):
        raise TypeError(
            f"Retriever returned an invalid result for {case_id!r}: "
            "expected a list of chunks."
        )

    validated: list[dict[str, Any]] = []

    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise TypeError(
                f"Invalid chunk #{index} for {case_id!r}: expected a dictionary."
            )

        chunk_id = chunk.get("chunk_id")
        source = chunk.get("source")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(
                f"Chunk #{index} for {case_id!r} is missing a valid 'chunk_id'."
            )

        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                f"Chunk #{index} for {case_id!r} is missing a valid 'source'."
            )

        validated.append(chunk)

    return validated


def evaluate_benchmark(
    benchmark_path: str | Path,
    adapter: Any | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate retrieval against benchmark cases.

    Relevance metrics are calculated only for cases with non-empty
    relevant_chunk_ids. Unlabelled cases receive diagnostic results only.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    path = Path(benchmark_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    benchmark = load_benchmark(path)

    if not benchmark:
        raise ValueError(f"Benchmark contains no cases: {path}")

    rag = adapter if adapter is not None else RAGAdapter(default_top_k=top_k)

    case_results: list[dict[str, Any]] = []
    labelled_precision: list[float] = []
    labelled_recall: list[float] = []
    labelled_mrr: list[float] = []
    duplicate_rates: list[float] = []
    source_hits: list[float] = []

    for case in benchmark:
        case_id = case["id"]
        question = case["question"]
        retrieved_chunks = _validate_retrieved_chunks(
            rag.retrieve(question, top_k=top_k), case_id
        )

        chunk_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]
        relevant_ids = set(case["relevant_chunk_ids"])
        relevant_sources = set(case["relevant_sources"])
        retrieved_sources = [chunk["source"] for chunk in retrieved_chunks]

        has_ground_truth = bool(relevant_ids)
        source_hit = bool(set(retrieved_sources) & relevant_sources)
        case_duplicate_rate = duplicate_rate(chunk_ids)

        result: dict[str, Any] = {
            "id": case_id,
            "question": question,
            "status": "evaluated" if has_ground_truth else "needs_ground_truth",
            "top_k": top_k,
            "retrieved_count": len(retrieved_chunks),
            "retrieved_chunks": retrieved_chunks,
            "retrieved_chunk_ids": chunk_ids,
            "retrieved_sources": retrieved_sources,
            "source_hits": sorted(set(retrieved_sources) & relevant_sources),
            "source_hit": source_hit,
            "duplicate_rate": case_duplicate_rate,
        }

        duplicate_rates.append(case_duplicate_rate)
        source_hits.append(float(source_hit))

        if has_ground_truth:
            p_at_k = precision_at_k(chunk_ids, relevant_ids, top_k)
            r_at_k = recall_at_k(chunk_ids, relevant_ids, top_k)
            mrr = reciprocal_rank(chunk_ids, relevant_ids)

            metric_values = (p_at_k, r_at_k, mrr)
            if not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in metric_values
            ):
                raise ValueError(
                    f"Non-finite relevance metric produced for case {case_id!r}."
                )

            result["metrics"] = {
                f"precision@{top_k}": float(p_at_k),
                f"recall@{top_k}": float(r_at_k),
                "mrr": float(mrr),
            }

            labelled_precision.append(float(p_at_k))
            labelled_recall.append(float(r_at_k))
            labelled_mrr.append(float(mrr))
        else:
            result["metrics"] = None

        case_results.append(result)

    labelled_count = len(labelled_precision)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_path": str(path),
        "top_k": top_k,
        "total_cases": len(benchmark),
        "labelled_cases": labelled_count,
        "unlabelled_cases": len(benchmark) - labelled_count,
        "aggregate_metrics": {
            f"precision@{top_k}": mean(labelled_precision),
            f"recall@{top_k}": mean(labelled_recall),
            "mrr": mean(labelled_mrr),
            "source_hit_rate": mean(source_hits),
            "duplicate_rate": mean(duplicate_rates),
        },
        "metrics_note": (
            "Relevance metrics are averaged over labelled cases only. "
            "Source hit rate and duplicate rate cover all benchmark cases. "
            "If there are no labelled cases, relevance aggregates are 0.0 "
            "placeholders and must not be interpreted as measured scores. "
            "Ground-truth labels should be manually reviewed."
        ),
        "results": case_results,
    }


def print_summary(report: dict[str, Any]) -> None:
    """Print a readable summary of the evaluation report."""
    top_k = report["top_k"]
    metrics = report["aggregate_metrics"]

    print("\n" + "=" * 68)
    print("AI-7 | RAG RETRIEVAL EVALUATION")
    print("=" * 68)
    print(f"Benchmark cases : {report['total_cases']}")
    print(f"Labelled cases  : {report['labelled_cases']}")
    print(f"Unlabelled cases: {report['unlabelled_cases']}")
    print(f"Retrieval top-k : {top_k}")
    print("-" * 68)

    if report["labelled_cases"] == 0:
        print("WARNING: No labelled cases. Relevance metrics are not meaningful.")
    else:
        print(f"Precision@{top_k:<2}    : {metrics[f'precision@{top_k}']:.4f}")
        print(f"Recall@{top_k:<2}       : {metrics[f'recall@{top_k}']:.4f}")
        print(f"MRR             : {metrics['mrr']:.4f}")

    print(f"Source hit rate  : {metrics['source_hit_rate']:.4f}")
    print(f"Duplicate rate   : {metrics['duplicate_rate']:.4f}")
    print("-" * 68)

    print("PER-QUESTION RESULTS")
    for result in report["results"]:
        print(f"\n[{result['id']}] {result['question']}")
        print(f"  Status       : {result['status']}")
        print(f"  Source hit   : {'Yes' if result['source_hit'] else 'No'}")
        print(f"  Retrieved    : {result['retrieved_count']} chunks")

        case_metrics = result["metrics"]
        if case_metrics is not None:
            print(
                f"  Precision@{top_k}: "
                f"{case_metrics[f'precision@{top_k}']:.4f}"
            )
            print(
                f"  Recall@{top_k}   : "
                f"{case_metrics[f'recall@{top_k}']:.4f}"
            )
            print(f"  MRR          : {case_metrics['mrr']:.4f}")

    print("\n" + "=" * 68)


def main() -> int:
    """Parse command-line arguments, evaluate, and save the report."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval quality against a benchmark."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help="Path to benchmark JSON (default: data/benchmark.json).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the JSON report.",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be a positive integer.")

    print("Starting AI-7 retrieval evaluation...")
    print(f"Benchmark: {args.benchmark}")
    print(f"Top-k: {args.top_k}")

    try:
        report = evaluate_benchmark(
            benchmark_path=args.benchmark,
            top_k=args.top_k,
        )

        output_path = args.output.expanduser()
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")

        print_summary(report)
        print(f"\nFull JSON report saved to: {output_path}")

        if report["unlabelled_cases"] > 0:
            print(
                "\nWARNING: Some benchmark cases have no ground-truth labels. "
                "Review them before relying on aggregate relevance metrics."
            )

        return 0

    except KeyboardInterrupt:
        print("\nEvaluation cancelled by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(
            f"\nERROR: Evaluation failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "Check the benchmark path, retrieval adapter, database, and "
            "dependencies. See the traceback by running the module in a "
            "debugger if the cause is unclear.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
