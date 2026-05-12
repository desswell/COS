import csv
from pathlib import Path

import numpy as np
import pytest

from src.utils.io import append_csv_row, write_json, read_json, save_npy, load_npy


def test_append_csv_row_creates_file_with_header(tmp_path):
    p = tmp_path / "out.csv"
    append_csv_row(p, {"epoch": 1, "loss": 0.5})
    assert p.exists()
    rows = list(csv.DictReader(p.open()))
    assert rows == [{"epoch": "1", "loss": "0.5"}]


def test_append_csv_row_appends_without_duplicating_header(tmp_path):
    p = tmp_path / "out.csv"
    append_csv_row(p, {"epoch": 1, "loss": 0.5})
    append_csv_row(p, {"epoch": 2, "loss": 0.3})
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 2
    assert rows[1]["epoch"] == "2"


def test_append_csv_row_rejects_mismatched_keys(tmp_path):
    p = tmp_path / "out.csv"
    append_csv_row(p, {"a": 1, "b": 2})
    with pytest.raises(ValueError, match="header mismatch"):
        append_csv_row(p, {"a": 3, "c": 4})


def test_write_and_read_json(tmp_path):
    p = tmp_path / "x.json"
    data = {"a": 1, "b": [1, 2, 3], "c": "hello"}
    write_json(p, data)
    assert read_json(p) == data


def test_save_and_load_npy(tmp_path):
    p = tmp_path / "arr.npy"
    arr = np.arange(12).reshape(3, 4).astype(np.float32)
    save_npy(p, arr)
    out = load_npy(p)
    assert np.array_equal(arr, out)
