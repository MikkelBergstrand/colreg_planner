"""scripts/tools/plot_runtime.py

Plot planner replan runtime against the number of vessels, using the
parallel-worker (workers=auto) samples from outputs/benchmarks/raw_samples.csv.
VO rollout timings are ignored.

Usage
-----
    python scripts/tools/plot_runtime.py
    python scripts/tools/plot_runtime.py --csv outputs/benchmarks/raw_samples.csv \
        --out outputs/benchmarks/runtime_vs_vessels.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SERIES_COLORS = {"2": "#2a78d6", "3": "#eb6834"}  # depth -> colour
TEXT = "#3d3d3a"
MUTED = "#8a8980"
GRID = "#e4e3dd"


def load_replan_times(path: Path) -> dict[str, dict[int, list[float]]]:
    """Return {depth: {n_vessels: [seconds, ...]}} for workers=auto replans."""
    data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["benchmark"] != "planner" or row["workers"] != "auto":
                continue
            if row["kind"] != "replan":
                continue
            n_vessels = int(row["n_others"]) + 1  # include ego
            data[row["depth"]][n_vessels].append(float(row["seconds"]))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument("--csv", type=Path, default=Path("outputs/benchmarks/raw_samples.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/benchmarks/runtime_vs_vessels.png"))
    args = parser.parse_args()

    data = load_replan_times(args.csv)

    plt.rcParams.update({"font.size": 11, "text.color": TEXT,
                         "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for depth in sorted(data):
        by_n = data[depth]
        ns = np.array(sorted(by_n))
        samples = [np.asarray(by_n[n]) for n in ns]
        median = np.array([np.median(s) for s in samples])
        p10 = np.array([np.percentile(s, 10) for s in samples])
        p90 = np.array([np.percentile(s, 90) for s in samples])
        color = SERIES_COLORS.get(depth, "#5c5b55")

        ax.fill_between(ns, p10, p90, color=color, alpha=0.15, linewidth=0)
        ax.plot(ns, median, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, label=f"Depth {depth}")
        ax.annotate(f"Depth {depth}", (ns[-1], median[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=TEXT)

    ax.set_xlabel("Number of vessels (incl. ego)")
    ax.set_ylabel("Replan time [s]")
    ax.set_title("Planner replan runtime (parallel workers)", loc="left", color=TEXT)
    ax.set_xticks(ns)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, loc="upper left", title="Median, shaded p10–p90",
              title_fontsize=9, alignment="left")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
