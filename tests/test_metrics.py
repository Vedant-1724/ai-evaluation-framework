import pytest

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    duplicate_rate,
    mean,
)


def test_precision_at_k():
    assert precision_at_k(["A", "B", "C"], {"A", "C"}, k=3) == pytest.approx(2 / 3)


def test_recall_at_k():
    assert recall_at_k(["A", "B", "C"], {"A", "C", "D"}, k=3) == pytest.approx(2 / 3)


def test_reciprocal_rank():
    assert reciprocal_rank(["A", "B", "C"], {"C"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_match():
    assert reciprocal_rank(["A", "B"], {"C"}) == 0.0


def test_duplicate_rate():
    assert duplicate_rate(["A", "B", "A", "C", "B"]) == pytest.approx(2 / 5)


def test_mean():
    assert mean([1, 2, 3, 4]) == pytest.approx(2.5)


def test_empty_inputs():
    assert precision_at_k([], {"A"}, k=5) == 0.0
    assert recall_at_k([], {"A"}) == 0.0
    assert reciprocal_rank([], {"A"}) == 0.0
    assert duplicate_rate([]) == 0.0
    assert mean([]) == 0.0


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["A"], {"A"}, k=0)