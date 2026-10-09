"""Build a nominal track library from raw Kystverket AIS parquet files.

The output HDF5 file can be loaded at runtime by NominalRetriever to provide
preferred-velocity sequences for the VO-based predictor.

Usage
-----
    python scripts/build_nominal_library.py \\
        --data-dir "data/Moss Horten Feb 2026 AIS" \\
        --out data/nominal_library.h5

    # Limit to first 3 daily files (smoke-test):
    python scripts/build_nominal_library.py \\
        --data-dir "data/Moss Horten Feb 2026 AIS" \\
        --out data/nominal_test.h5 \\
        --days 3

    # Custom parameters:
    python scripts/build_nominal_library.py \\
        --data-dir "data/Moss Horten Feb 2026 AIS" \\
        --out data/nominal_library.h5 \\
        --dt 10 --l 20 --H 60 --stride 5 --min-sog 1.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


from colregs_planner.nominal.indexer import TrackIndexer
from colregs_planner.nominal.library import LibraryConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build nominal track library from raw AIS parquet files."
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Directory containing *.parquet AIS files.",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output HDF5 path (e.g. data/nominal_library.h5).",
    )
    parser.add_argument("--dt", type=float, default=2.0,
                        help="Resample interval [s] (default 2.0).")
    parser.add_argument("--l", type=int, default=60,
                        help="Lookback steps for DTW matching (default 60).")
    parser.add_argument("--H", type=int, default=300,
                        help="Prediction horizon steps to store (default 300).")
    parser.add_argument("--stride", type=int, default=5,
                        help="Sliding-window stride (default 5).")
    parser.add_argument("--min-sog", type=float, default=1.5,
                        help="Minimum SOG [kn] to include a window step (default 1.5).")
    parser.add_argument("--gap", type=float, default=600.0,
                        help="Voyage gap threshold [s] for segmentation (default 600).")
    parser.add_argument("--days", type=int, default=None,
                        help="Limit to first N parquet files (for testing).")
    args = parser.parse_args()

    parquet_files = sorted(args.data_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"ERROR: no parquet files found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)
    if args.days is not None:
        parquet_files = parquet_files[: args.days]

    print(f"Found {len(parquet_files)} parquet file(s).")
    print(
        f"Config: dt={args.dt}s  l={args.l}  H={args.H}  "
        f"stride={args.stride}  min_sog={args.min_sog} kn  gap={args.gap}s"
    )
    print(f"Output: {args.out}\n")

    config = LibraryConfig(l=args.l, H=args.H, stride=args.stride, dt=args.dt)
    indexer = TrackIndexer(
        config,
        min_sog_knots=args.min_sog,
        gap_threshold_s=args.gap,
    )

    library = indexer.fit_from_parquet(parquet_files, verbose=True)

    print(f"\nLibrary size: {len(library):,} subtrajectories")
    print(f"  pos_match shape: {library.pos_match.shape}")
    print(f"  vel_fwd   shape: {library.vel_fwd.shape}")

    library.save(args.out)
    size_mb = args.out.stat().st_size / 1e6
    print(f"\nSaved to {args.out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
