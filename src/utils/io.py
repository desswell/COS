"""Small IO helpers used across training and aggregation scripts."""
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def append_csv_row(path: str | Path, row: Mapping[str, Any]) -> None:
    """Append a row to a CSV, writing the header on first write.

    Raises ValueError if the existing file's header does not match the row's keys.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    file_exists = path.exists()

    if file_exists:
        with path.open("r", newline="") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header and existing_header != fieldnames:
            raise ValueError(
                f"header mismatch: file has {existing_header}, row has {fieldnames}"
            )

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_npy(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_npy(path: str | Path) -> np.ndarray:
    return np.load(path)
