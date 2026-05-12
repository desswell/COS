"""Download and extract UrbanSound8K.

Run as: python -m src.data.download

Two sources are supported:
- Zenodo tarball (the official one at https://zenodo.org/records/1203745). Tried
  first, but Zenodo aggressively blocks repeated requests from the same IP — if
  it returns 403 we fall back automatically.
- HuggingFace mirror (`danavery/urbansound8K`, CC BY-NC, identical contents). The
  fallback path. After download we re-emit the files in the original layout
  (`UrbanSound8K/audio/foldN/*.wav` + `UrbanSound8K/metadata/UrbanSound8K.csv`)
  so the rest of the pipeline does not need to know which source was used.
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
HF_REPO_ID = "danavery/urbansound8K"
HF_CSV_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/UrbanSound8K.csv"
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


def _download_from_huggingface(dst_dir: Path) -> None:
    """Download UrbanSound8K from the HuggingFace mirror and re-emit in original layout.

    Uses `datasets.load_dataset("danavery/urbansound8K")` which fetches parquet
    shards (~7 GB) into the HF cache, then iterates over rows and writes each
    audio array back to disk as `audio/foldN/<slice_file_name>` (wav). The
    `UrbanSound8K.csv` metadata file is downloaded separately from the same repo
    so we get the official 10-fold splits.
    """
    from datasets import load_dataset, Audio

    dst = Path(dst_dir) / "UrbanSound8K"
    audio_root = dst / "audio"
    metadata_dir = dst / "metadata"
    metadata_csv = metadata_dir / "UrbanSound8K.csv"
    audio_root.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_csv.exists():
        print(f"[download] Fetching metadata CSV from {HF_CSV_URL}")
        urllib.request.urlretrieve(HF_CSV_URL, str(metadata_csv))

    print(f"[download] Loading dataset {HF_REPO_ID} from HuggingFace (~7 GB on first run)...")
    # decode=False keeps the raw bytes (wav-encoded) — we just write them to disk as-is,
    # which avoids the torchcodec / torchaudio decoder path entirely.
    ds = load_dataset(HF_REPO_ID, split="train").cast_column("audio", Audio(decode=False))

    print(f"[download] Re-emitting {len(ds)} files into {audio_root} ...")
    for i, row in enumerate(tqdm(ds, total=len(ds), desc="extract", unit="file")):
        slice_name = row["slice_file_name"]
        fold = int(row["fold"])
        audio = row["audio"]  # {'path': str, 'bytes': bytes} when decode=False
        fold_dir = audio_root / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        out_path = fold_dir / slice_name
        if not out_path.exists():
            data = audio.get("bytes")
            if data is None:
                # Some HF audio entries may store only a path to a cached file.
                src_path = audio.get("path")
                if src_path and Path(src_path).exists():
                    data = Path(src_path).read_bytes()
                else:
                    raise RuntimeError(f"No bytes or path for {slice_name}")
            out_path.write_bytes(data)
    print(f"[download] Done.")


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


def download(dst_dir: Path, source: str = "auto") -> None:
    """High-level entry: pick a source and run it.

    source: 'zenodo' | 'huggingface' | 'auto' (try Zenodo first, then HF on failure).
    """
    dst_dir = Path(dst_dir)
    metadata = dst_dir / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"
    if metadata.exists():
        print(f"[download] Already extracted at {metadata.parent.parent}, skipping.")
        return

    if source == "zenodo":
        download_and_extract(dst_dir)
        return
    if source == "huggingface":
        _download_from_huggingface(dst_dir)
        return

    # source == 'auto'
    try:
        download_and_extract(dst_dir)
        return
    except Exception as e:
        print(f"[download] Zenodo failed ({type(e).__name__}: {e}). Falling back to HuggingFace mirror.")
        # Clean up partial tarball from Zenodo attempt
        tar_path = dst_dir / "UrbanSound8K.tar.gz"
        if tar_path.exists():
            tar_path.unlink()
        _download_from_huggingface(dst_dir)


def main():
    parser = argparse.ArgumentParser(description="Download UrbanSound8K (Zenodo or HuggingFace mirror)")
    parser.add_argument("--dst", type=Path, default=Path("data"))
    parser.add_argument(
        "--source", default="auto", choices=["auto", "zenodo", "huggingface"],
        help="auto = try Zenodo first then HF (default); zenodo or huggingface to force one source",
    )
    parser.add_argument("--url", default=ZENODO_URL, help="Override Zenodo URL (only relevant for --source zenodo)")
    args = parser.parse_args()
    if args.source == "zenodo" and args.url != ZENODO_URL:
        download_and_extract(args.dst, url=args.url)
    else:
        download(args.dst, source=args.source)


if __name__ == "__main__":
    sys.exit(main())
