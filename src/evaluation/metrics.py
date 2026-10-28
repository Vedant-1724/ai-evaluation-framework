
from collections.abc import Iterable, Sequence


def precision_at_k(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int = 5,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not retrieved:
        return 0.0

    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)

    return hits / len(top_k)


def recall_at_k(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int | None = None,
) -> float:
    if not relevant:
        return 0.0

    if k is not None and k <= 0:
        raise ValueError("k must be greater than zero")

    selected = retrieved if k is None else retrieved[:k]
    hits = sum(1 for item in selected if item in relevant)

    return hits / len(relevant)


def reciprocal_rank(
    retrieved: Sequence[str],
    relevant: set[str],
) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank

    return 0.0


def duplicate_rate(retrieved: Sequence[str]) -> float:
    if not retrieved:
        return 0.0

    return (len(retrieved) - len(set(retrieved))) / len(retrieved)


def mean(values: Iterable[float]) -> float:
    values = list(values)

    if not values:
        return 0.0

    return sum(values) / len(values)
