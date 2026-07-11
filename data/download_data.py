"""Download the Heart Disease UCI dataset.

Two acquisition strategies are attempted, in order:

1. ``ucimlrepo`` package (official UCI helper, dataset id=45).
2. Direct HTTP download of ``processed.cleveland.data`` from the UCI archive.

The raw file is saved to ``data/heart_disease_raw.csv`` with proper headers.

Usage
-----
    python data/download_data.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

# Allow running as a plain script (add project root to path).
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import COLUMN_NAMES, RAW_DATA_PATH, DATA_DIR  # noqa: E402

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)


def _from_ucimlrepo() -> pd.DataFrame | None:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print("[download] ucimlrepo not installed, skipping.")
        return None
    try:
        print("[download] Fetching via ucimlrepo (id=45)...")
        ds = fetch_ucirepo(id=45)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        # Normalise column names to our canonical schema.
        df.columns = COLUMN_NAMES
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"[download] ucimlrepo failed: {exc}")
        return None


def _from_http() -> pd.DataFrame | None:
    try:
        import urllib.request

        print(f"[download] Fetching raw file from {UCI_URL} ...")
        with urllib.request.urlopen(UCI_URL, timeout=30) as resp:  # noqa: S310
            content = resp.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content), header=None, names=COLUMN_NAMES)
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"[download] HTTP download failed: {exc}")
        return None


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = _from_ucimlrepo()
    if df is None:
        df = _from_http()
    if df is None:
        raise SystemExit(
            "Could not obtain the dataset. Check your internet connection or "
            "manually place processed.cleveland.data into the data/ folder."
        )

    # The raw file uses '?' for missing values (mainly in ca/thal).
    df = df.replace("?", pd.NA)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"[download] Saved {len(df)} rows -> {RAW_DATA_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
