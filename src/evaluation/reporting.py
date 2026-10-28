
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def save_report(
    report: dict[str, Any],
    output_dir: str | Path = "reports",
    filename: str | None = None,
) -> Path:
    """Save an evaluation report as formatted JSON."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"evaluation_{timestamp}.json"

    if Path(filename).name != filename or not filename.endswith(".json"):
        raise ValueError("filename must be a simple .json filename")

    destination = output_path / filename

    with destination.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    return destination


def load_report(path: str | Path) -> dict[str, Any]:
    """Load a previously saved evaluation report."""

    report_path = Path(path)

    if not report_path.is_file():
        raise FileNotFoundError(f"Report not found: {report_path}")

    with report_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Report must contain a JSON object")

    return data
