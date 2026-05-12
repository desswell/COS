"""Download and extract UrbanSound8K from Zenodo.

Run as: python -m src.data.download
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm


ZENODO_URL = "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz"
# Official SHA-256 from Zenodo metadata (verify on first real download and
# update if Zenodo re-publishes the file).
EXPECTED_SHA256 = None  # set to actual hash if you want strict verification


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _download_with_progress(url: str, out_path: Path) -> None:
    """Stream-download with a tqdm progress bar."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with out_path.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="Downloading"
        ) as pbar:
            while chunk := resp.read(1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))


def download_and_extract(
    dst_dir: Path,
    url: str = ZENODO_URL,
    expected_sha256: str | None = EXPECTED_SHA256,
) -> None:
    """Download and extract UrbanSound8K. Idempotent: skips if metadata file already present.

    Layout after extraction: `dst_dir/UrbanSound8K/{audio, metadata}/...`
    """
    dst_dir = Path(dst_dir)
    metadata = dst_dir / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"
    if metadata.exists():
        print(f"[download] Already extracted at {metadata.parent.parent}, skipping.")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dst_dir / "UrbanSound8K.tar.gz"

    if not tar_path.exists():
        print(f"[download] Downloading from {url} -> {tar_path}")
        _download_with_progress(url, tar_path)
    else:
        print(f"[download] Found existing archive {tar_path}, skipping download.")

    if expected_sha256 is not None:
        print("[download] Verifying SHA-256...")
        actual = _sha256(tar_path)
        if actual != expected_sha256:
            raise RuntimeError(f"SHA-256 mismatch: expected {expected_sha256}, got {actual}")

    print(f"[download] Extracting {tar_path} -> {dst_dir}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(dst_dir)

    if not metadata.exists():
        raise RuntimeError(f"Extraction completed but metadata not found at {metadata}")
    print("[download] Done.")


def main():
    parser = argparse.ArgumentParser(description="Download UrbanSound8K from Zenodo")
    parser.add_argument("--dst", type=Path, default=Path("data"))
    parser.add_argument("--url", default=ZENODO_URL)
    args = parser.parse_args()
    download_and_extract(args.dst, url=args.url)


if __name__ == "__main__":
    sys.exit(main())
