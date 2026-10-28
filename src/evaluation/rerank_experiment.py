import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from sentence_transformers import CrossEncoder

from src.retriever import retrieve
from src.evaluation.benchmark import load_benchmark
from src.evaluation.metrics import (
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    duplicate_rate,
)


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate cross-encoder reranking."
    )
    parser.add_argument(
        "--benchmark",
        default="data/benchmark_audited_v1.json",
    )
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default="reports/reranking_experiment",
    )
    args = parser.parse_args()

    if args.top_k < 1 or args.candidate_k < args.top_k:
        parser.error("Require candidate-k >= top-k >= 1.")

    benchmark = load_benchmark(args.benchmark)

    labelled_cases = [
        case for case in benchmark
        if case["status"] == "labelled"
    ]

    if not labelled_cases:
        raise ValueError("Benchmark contains no labelled cases.")

    print(f"Loading reranker: {MODEL_NAME}")
    reranker = CrossEncoder(MODEL_NAME)

    results = []

    for index, case in enumerate(benchmark, start=1):
        print(f"[{index}/{len(benchmark)}] {case['id']}")

        candidates = retrieve(
            case["question"],
            top_k=args.candidate_k,
        )

        pairs = [
            (case["question"], chunk["content"])
            for chunk in candidates
        ]

        scores = reranker.predict(pairs) if pairs else []

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        reranked = [
            chunk for chunk, _ in ranked[:args.top_k]
        ]
        retrieved_ids = [
            chunk["chunk_id"] for chunk in reranked
        ]

        relevant_ids = set(case["relevant_chunk_ids"])
        labelled = case["status"] == "labelled"

        metrics = {
            f"precision@{args.top_k}": (
                precision_at_k(
                    retrieved_ids, relevant_ids, args.top_k
                ) if labelled else None
            ),
            f"recall@{args.top_k}": (
                recall_at_k(
                    retrieved_ids, relevant_ids, args.top_k
                ) if labelled else None
            ),
            "mrr": (
                reciprocal_rank(retrieved_ids, relevant_ids)
                if labelled else None
            ),
            "duplicate_rate": duplicate_rate(retrieved_ids),
            "source_hit": int(
                any(
                    chunk["source"] in case["relevant_sources"]
                    for chunk in reranked
                )
            ),
        }

        results.append({
            "id": case["id"],
            "question": case["question"],
            "status": case["status"],
            "retrieved_chunk_ids": retrieved_ids,
            "metrics": metrics,
            "reranker_scores": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "score": float(score),
                }
                for chunk, score in ranked
            ],
        })

    def aggregate(metric_name):
        values = [
            result["metrics"][metric_name]
            for result in results
            if result["metrics"][metric_name] is not None
        ]
        return mean(values)

    output = {
        "experiment": "cross_encoder_reranking",
        "model": MODEL_NAME,
        "benchmark": args.benchmark,
        "candidate_k": args.candidate_k,
        "top_k": args.top_k,
        "case_count": len(results),
        "labelled_case_count": len(labelled_cases),
        "aggregate_metrics": {
            f"precision@{args.top_k}": aggregate(
                f"precision@{args.top_k}"
            ),
            f"recall@{args.top_k}": aggregate(
                f"recall@{args.top_k}"
            ),
            "mrr": aggregate("mrr"),
            "source_hit_rate": aggregate("source_hit"),
            "duplicate_rate": aggregate("duplicate_rate"),
        },
        "metrics_note": (
            "Relevance metrics are averaged over labelled cases only. "
            "Source hit rate and duplicate rate cover all cases. "
            "This evaluates retrieval ranking, not generated-answer quality."
        ),
        "results": results,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"reranking_{timestamp}.json"

    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\nAggregate metrics:")
    print(json.dumps(output["aggregate_metrics"], indent=2))
    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()
