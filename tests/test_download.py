import hashlib
import io
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.data.download import download_and_extract, _sha256


def _make_fake_tarball(tmp_path) -> Path:
    """Create a tarball containing UrbanSound8K/metadata/UrbanSound8K.csv with 1 line."""
    inner = tmp_path / "inner"
    (inner / "UrbanSound8K" / "metadata").mkdir(parents=True)
    (inner / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv").write_text(
        "slice_file_name,fsID,start,end,salience,fold,classID,class\n"
        "x.wav,0,0.0,4.0,1,1,0,air_conditioner\n"
    )
    tar_path = tmp_path / "fake.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(inner / "UrbanSound8K", arcname="UrbanSound8K")
    return tar_path


def test_sha256_computes_correctly(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    assert _sha256(p) == hashlib.sha256(b"hello").hexdigest()


def test_download_and_extract_extracts_archive(tmp_path, monkeypatch):
    tar_path = _make_fake_tarball(tmp_path)
    dst = tmp_path / "data"

    def fake_download(url, out_path):
        out_path.write_bytes(tar_path.read_bytes())

    with patch("src.data.download._download_with_progress", side_effect=fake_download):
        download_and_extract(dst, url="http://fake", expected_sha256=None)

    assert (dst / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv").exists()


def test_download_and_extract_idempotent(tmp_path):
    dst = tmp_path / "data"
    (dst / "UrbanSound8K" / "metadata").mkdir(parents=True)
    (dst / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv").write_text("ok")

    # If already present, must NOT try to download
    with patch("src.data.download._download_with_progress") as mock_dl:
        download_and_extract(dst, url="http://fake", expected_sha256=None)
        mock_dl.assert_not_called()
