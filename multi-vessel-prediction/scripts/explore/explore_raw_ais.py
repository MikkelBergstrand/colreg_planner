"""scripts/explore/explore_raw_ais.py

Quick console-based exploration of the raw AIS parquet files from Kystverket.

Run this first to understand the data schema and quality before processing.

Usage
-----
    python scripts/explore/explore_raw_ais.py                          # first day file
    python scripts/explore/explore_raw_ais.py --file hais_2025-07-15   # specific day
    python scripts/explore/explore_raw_ais.py --all                    # aggregate all 31 files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "Moss Horten July 2025 AIS"

_AIS_NAV_STATUS = {
    0: "Under way (engine)",
    1: "At anchor",
    2: "Not under command",
    3: "Restricted manoeuvrability",
    4: "Constrained by draught",
    5: "Moored",
    6: "Aground",
    7: "Engaged in fishing",
    8: "Under way (sailing)",
    15: "Undefined",
}

_AIS_MSG_TYPE = {
    1: "Class A pos (scheduled)",
    2: "Class A pos (assigned)",
    3: "Class A pos (special)",
    5: "Static & voyage",
    18: "Class B pos",
    21: "Aid-to-navigation",
    24: "Class B static",
    27: "Long-range AIS",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_file(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["sog"] = pd.to_numeric(df["speed_over_ground"], errors="coerce")
    df["cog"] = pd.to_numeric(df["course_over_ground"], errors="coerce")
    # AIS not-available sentinels
    df.loc[df["cog"] >= 360.0, "cog"] = float("nan")
    df.loc[df["true_heading"] == 511, "true_heading"] = float("nan")
    df.loc[df["rate_of_turn"] == -128, "rate_of_turn"] = float("nan")
    return df


def _ascii_histogram(values: np.ndarray, bins: int = 12, width: int = 40) -> str:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return "  (no finite values)"
    counts, edges = np.histogram(finite, bins=bins)
    max_count = counts.max() if counts.max() > 0 else 1
    lines = []
    for lo, hi, c in zip(edges[:-1], edges[1:], counts):
        bar = "#" * int(c / max_count * width)
        lines.append(f"  [{lo:7.1f}, {hi:7.1f}) | {bar:<{width}} {c:>7,}")
    return "\n".join(lines)


def _sep(title: str, total: int = 70) -> str:
    return f"\n-- {title} " + "-" * max(0, total - len(title) - 4)


# ---------------------------------------------------------------------------
# Single-file report
# ---------------------------------------------------------------------------

def report_file(df: pd.DataFrame, fname: str) -> None:
    print("=" * 70)
    print(f"FILE: {fname}")
    print(f"Rows: {len(df):,}    Columns: {len(df.columns)}")
    print("=" * 70)

    print(_sep("SCHEMA"))
    for col in df.columns:
        dt = df[col].dtype
        null_pct = 100 * df[col].isna().sum() / max(len(df), 1)
        uniq = df[col].nunique(dropna=False)
        print(f"  {col:<26} {str(dt):<20} null={null_pct:5.1f}%  unique={uniq:,}")

    display_cols = [c for c in df.columns if c not in ("geometry", "speed_over_ground", "course_over_ground")]
    print(_sep("SAMPLE ROWS (first 5)"))
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df[display_cols].head(5).to_string(index=False))

    print(_sep("NUMERIC COLUMN STATISTICS"))
    num_cols = ["sog", "cog", "true_heading", "rate_of_turn", "latitude", "longitude"]
    for col in num_cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            print(f"  {col:<20} all NaN")
            continue
        print(
            f"  {col:<20} n={len(s):>7,}  min={s.min():8.2f}  mean={s.mean():8.2f}"
            f"  max={s.max():8.2f}  std={s.std():7.2f}"
        )

    print(_sep("SOG HISTOGRAM (knots)"))
    sog = df["sog"].dropna().values
    print(_ascii_histogram(sog[sog <= 50], bins=10))
    n_outlier = int((sog > 50).sum())
    if n_outlier:
        print(f"  + {n_outlier:,} rows with SOG > 50 kn (likely erroneous)")

    lat = df["latitude"].dropna()
    lon = df["longitude"].dropna()
    print(_sep("GEOGRAPHIC BOUNDS"))
    print(f"  Latitude:  {lat.min():.5f} -> {lat.max():.5f} N")
    print(f"  Longitude: {lon.min():.5f} -> {lon.max():.5f} E")

    print(_sep("TIME RANGE (UTC)"))
    print(f"  {df['date_time_utc'].min()} -> {df['date_time_utc'].max()}")

    print(_sep("NAVIGATION STATUS"))
    for code, cnt in df["status"].value_counts().items():
        label = _AIS_NAV_STATUS.get(int(code), f"code {code}")
        print(f"  {int(code):2d}  {label:<35} {cnt:>7,}")

    print(_sep("AIS MESSAGE TYPE"))
    for code, cnt in df["msg_type"].value_counts().items():
        label = _AIS_MSG_TYPE.get(int(code), f"type {code}")
        print(f"  {int(code):2d}  {label:<35} {cnt:>7,}")

    print(_sep("AIS CLASS"))
    for cls, cnt in df["ais_class"].value_counts().items():
        print(f"  {cls}  {cnt:>7,}")

    print(_sep("TOP 20 MMSI BY MESSAGE COUNT"))
    top = df.groupby("mmsi").size().sort_values(ascending=False).head(20)
    for mmsi, cnt in top.items():
        sub = df[df["mmsi"] == mmsi]
        sog_v = sub["sog"].dropna()
        sog_str = f"SOG {sog_v.min():.1f}--{sog_v.max():.1f} kn" if len(sog_v) else ""
        print(f"  {mmsi:>10}  {cnt:>6,} msgs   {sog_str}")

    print()


# ---------------------------------------------------------------------------
# Aggregate report (--all)
# ---------------------------------------------------------------------------

def report_all(files: list[Path]) -> None:
    print("=" * 70)
    print(f"AGGREGATE REPORT -- all {len(files)} daily files")
    print("=" * 70)

    total_rows = 0
    all_mmsi: set[int] = set()
    t_min: pd.Timestamp | None = None
    t_max: pd.Timestamp | None = None
    sog_list: list[np.ndarray] = []
    status_counts: dict[int, int] = {}

    for i, f in enumerate(files):
        print(f"\r  Loading {i + 1}/{len(files)}: {f.name} ...", end="", flush=True)
        df = _load_file(f)
        total_rows += len(df)
        all_mmsi |= set(df["mmsi"].unique())
        fmin, fmax = df["date_time_utc"].min(), df["date_time_utc"].max()
        t_min = fmin if t_min is None else min(t_min, fmin)
        t_max = fmax if t_max is None else max(t_max, fmax)
        sog_list.append(df["sog"].dropna().values)
        for code, cnt in df["status"].value_counts().items():
            status_counts[int(code)] = status_counts.get(int(code), 0) + cnt

    print()
    sog_all = np.concatenate(sog_list)

    print(f"\n  Total rows         : {total_rows:,}")
    print(f"  Unique MMSI        : {len(all_mmsi):,}")
    print(f"  Time span          : {t_min} -> {t_max}")
    if t_min and t_max:
        hours = (t_max - t_min).total_seconds() / 3600
        print(f"  Duration           : {hours:.1f} hours")
    print()
    print("  SOG distribution (all files, <= 50 kn):")
    valid_sog = sog_all[sog_all <= 50]
    print(_ascii_histogram(valid_sog, bins=10))
    n_outlier = int((sog_all > 50).sum())
    if n_outlier:
        print(f"  + {n_outlier:,} rows with SOG > 50 kn")

    print("\n  Navigation status breakdown:")
    for code, cnt in sorted(status_counts.items()):
        label = _AIS_NAV_STATUS.get(code, f"code {code}")
        pct = 100 * cnt / total_rows
        print(f"    {code:2d}  {label:<35} {cnt:>8,}  ({pct:.1f}%)")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Explore raw AIS parquet files.")
    parser.add_argument(
        "--file",
        default=None,
        help="Parquet file name or stem (default: first file in directory)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Load all 31 daily files and print aggregate stats",
    )
    args = parser.parse_args()

    files = sorted(_DATA_DIR.glob("*.parquet"))
    if not files:
        print(f"No parquet files found in {_DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    if args.all:
        report_all(files)
        return

    if args.file:
        stem = args.file.replace(".snappy.parquet", "").replace(".parquet", "")
        candidates = [f for f in files if stem in f.name]
        if not candidates:
            print(f"No file matching '{args.file}' found in {_DATA_DIR}", file=sys.stderr)
            sys.exit(1)
        target = candidates[0]
    else:
        target = files[0]

    print(f"Loading {target} ...")
    df = _load_file(target)
    report_file(df, target.name)


if __name__ == "__main__":
    main()
