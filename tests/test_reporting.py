
import json

import pytest

from src.evaluation.reporting import load_report, save_report


def test_save_and_load_report(tmp_path):
    report = {
        "total_cases": 2,
        "aggregate_metrics": {"mrr": 0.75},
        "results": [],
    }

    saved_path = save_report(report, output_dir=tmp_path)
    loaded = load_report(saved_path)

    assert loaded == report
    assert saved_path.exists()


def test_save_report_with_custom_filename(tmp_path):
    report = {"total_cases": 1}

    saved_path = save_report(
        report,
        output_dir=tmp_path,
        filename="test_report.json",
    )

    assert saved_path.name == "test_report.json"
    assert json.loads(saved_path.read_text(encoding="utf-8")) == report


def test_invalid_filename_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_report({}, output_dir=tmp_path, filename="../report.json")


def test_missing_report_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_report(tmp_path / "missing.json")


def test_non_object_report_is_rejected(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError):
        load_report(path)
