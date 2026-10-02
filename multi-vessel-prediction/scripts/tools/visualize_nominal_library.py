"""scripts/tools/visualize_nominal_library.py

Visualize and sanity-check a nominal track library.

Creates a three-panel figure:
  Left   — hexbin density of all subtrajectory end-positions in lat/lon
  Centre — random sample of full l-step tracks (NED-local, arrows show direction)
  Right  — histogram of mean SOG per track

Usage
-----
    python scripts/tools/visualize_nominal_library.py --library data/nominal_library.h5
    python scripts/tools/visualize_nominal_library.py --library data/nominal_library.h5 --samples 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


from src.nominal.library import TrackLibrary

_ROOT = Path(__file__).resolve().parents[2]


def _utm_to_latlon(north: np.ndarray, east: np.ndarray):
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    lon, lat = t.transform(east, north)
    return lat, lon


def _print_stats(lib: TrackLibrary) -> None:
    cfg = lib.config
    N = len(lib)
    pos = lib.pos_match          # (N, l, 2)
    vel = lib.vel_fwd            # (N, H+1, 2)
    speeds = np.linalg.norm(vel[:, 0, :], axis=1)  # speed at first fwd step

    end_pos = pos[:, -1, :]
    lat, lon = _utm_to_latlon(end_pos[:, 0], end_pos[:, 1])

    print(f"\n{'═' * 52}")
    print(f"  Library: {N:,} subtrajectories")
    print(f"  Config : l={cfg.l}  H={cfg.H}  stride={cfg.stride}  dt={cfg.dt}s")
    print(f"  Window : {cfg.l * cfg.dt:.0f}s lookback  +  {cfg.H * cfg.dt:.0f}s horizon")
    print(f"  Lat    : {lat.min():.4f} – {lat.max():.4f} °N")
    print(f"  Lon    : {lon.min():.4f} – {lon.max():.4f} °E")
    print(f"  Speed  : {speeds.min() * 1.944:.1f} – {speeds.max() * 1.944:.1f} kn"
          f"  (mean {speeds.mean() * 1.944:.1f} kn)")
    print(f"{'═' * 52}\n")


def visualize(lib: TrackLibrary, n_samples: int = 300, seed: int = 42) -> None:
    _print_stats(lib)

    cfg = lib.config
    N = len(lib)
    rng = np.random.default_rng(seed)

    # ---- geographic positions (end of each subtrajectory = current pos) ----
    end_pos = lib.pos_match[:, -1, :]           # (N, 2) UTM [north, east]
    lat, lon = _utm_to_latlon(end_pos[:, 0], end_pos[:, 1])

    # ---- sample tracks ----
    idx = rng.choice(N, size=min(n_samples, N), replace=False)
    sample_pos = lib.pos_match[idx].astype(np.float64)   # (S, l, 2)
    sample_vel = lib.vel_fwd[idx].astype(np.float64)     # (S, H+1, 2)

    # Speeds: mean SOG per track in knots
    speeds_ms = np.linalg.norm(lib.vel_fwd[:, :, :], axis=2).mean(axis=1)  # (N,)
    speeds_kn = speeds_ms * 1.944

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Nominal track library  —  {N:,} subtrajectories  "
        f"(l={cfg.l}, H={cfg.H}, dt={cfg.dt:.0f}s)",
        fontsize=12,
    )

    # --- Panel 1: geographic density hexbin ---
    ax = axes[0]
    hb = ax.hexbin(lon, lat, gridsize=60, cmap="YlOrRd", mincnt=1)
    fig.colorbar(hb, ax=ax, label="subtrajectory count")
    ax.set_xlabel("Longitude [°E]")
    ax.set_ylabel("Latitude [°N]")
    ax.set_title("Geographic density (end positions)")
    ax.set_aspect("auto")

    # --- Panel 2: sample tracks in local NED ---
    ax = axes[1]
    # Centre each track on its own start so they can be compared by shape
    S = len(sample_pos)
    alpha = max(0.03, min(0.3, 20.0 / S))
    for trk in sample_pos:
        rel = trk - trk[0]              # (l, 2) relative to start
        ax.plot(rel[:, 1], rel[:, 0], color="steelblue", lw=0.6, alpha=alpha)
        # Arrow at end showing direction
        if len(rel) >= 2:
            dy = rel[-1, 0] - rel[-2, 0]
            dx = rel[-1, 1] - rel[-2, 1]
            ax.annotate(
                "", xy=(rel[-1, 1], rel[-1, 0]),
                xytext=(rel[-1, 1] - dx, rel[-1, 0] - dy),
                arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.6, alpha=alpha * 2),
            )
    ax.set_xlabel("East offset [m]")
    ax.set_ylabel("North offset [m]")
    ax.set_title(f"{S} sample tracks (relative to start)")
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.4, alpha=0.3)
    ax.axvline(0, color="k", lw=0.4, alpha=0.3)

    # Also plot a few forward velocity sequences in a separate inset
    ax2_inset = ax.inset_axes([0.62, 0.62, 0.36, 0.36])
    fwd_steps = np.arange(cfg.H + 1) * cfg.dt
    for sv in sample_vel[:min(50, S)]:
        sp_kn = np.linalg.norm(sv, axis=1) * 1.944
        ax2_inset.plot(fwd_steps, sp_kn, color="darkorange", lw=0.5, alpha=0.3)
    ax2_inset.set_xlabel("fwd [s]", fontsize=7)
    ax2_inset.set_ylabel("kn", fontsize=7)
    ax2_inset.tick_params(labelsize=6)
    ax2_inset.set_title("SOG fwd", fontsize=7)

    # --- Panel 3: speed distribution histogram ---
    ax = axes[2]
    ax.hist(speeds_kn, bins=60, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(speeds_kn.mean(), color="crimson", lw=1.5, label=f"mean {speeds_kn.mean():.1f} kn")
    ax.set_xlabel("Mean SOG [kn]")
    ax.set_ylabel("Number of subtrajectories")
    ax.set_title("Speed distribution")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a nominal track library (.h5)."
    )
    parser.add_argument(
        "--library", type=Path,
        default=_ROOT / "data" / "nominal_library.h5",
        help="Path to the HDF5 library file.",
    )
    parser.add_argument(
        "--samples", type=int, default=300,
        help="Number of random tracks to draw in the centre panel (default 300).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sample selection (default 42).",
    )
    args = parser.parse_args()

    if not args.library.exists():
        print(f"ERROR: library not found: {args.library}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.library} ...")
    lib = TrackLibrary.load(args.library)
    visualize(lib, n_samples=args.samples, seed=args.seed)


if __name__ == "__main__":
    main()
