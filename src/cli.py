
import argparse
import json
import sys
from pathlib import Path

from src.evaluation.reporting import save_report
from src.evaluation.runner import evaluate_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a RAG system against a benchmark."
    )

    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/benchmark.json"),
        help="Path to the benchmark JSON file.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory where evaluation reports will be saved.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    try:
        report = evaluate_benchmark(
            benchmark_path=args.benchmark,
            top_k=args.top_k,
        )

        report_path = save_report(
            report,
            output_dir=args.output_dir,
        )

    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    print("\nEvaluation completed successfully.")
    print(f"Benchmark cases: {report['total_cases']}")
    print(f"Labelled cases: {report['labelled_cases']}")
    print(f"Unlabelled cases: {report['unlabelled_cases']}")
    print(f"Top-K: {report['top_k']}")
    print(f"Report saved to: {report_path}")

    print("\nAggregate metrics:")
    print(json.dumps(report["aggregate_metrics"], indent=2))

    if report["labelled_cases"] == 0:
        print(
            "\nNOTE: No ground-truth chunk IDs have been labelled yet. "
            "Relevance metrics are placeholders, not measured performance."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
