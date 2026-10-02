"""
Two overview plots of raw AIS data — Moss–Horten region:
  1. Position density heatmap  – where do vessels spend time?
  2. Velocity vector field     – what heading are vessels on at each position?

Both plots use OpenStreetMap tile backgrounds via contextily.
Outputs saved to: outputs/plots/<label>/

Usage:
    python plot_ais_overview.py                  # runs all configured datasets
    python plot_ais_overview.py july2025         # single dataset by label
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import contextily as ctx
from pyproj import Transformer

# ── Paths ───────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "plots"

# ── Datasets ─────────────────────────────────────────────────────────────────────
DATASETS = {
    "july2025": {
        "dir":   PROJECT_ROOT / "data" / "Moss Horten July 2025 AIS",
        "label": "July 2025",
    },
    "feb2026": {
        "dir":   PROJECT_ROOT / "data" / "Moss Horten Feb 2026 AIS",
        "label": "February 2026",
    },
}

# ── Geographic bounds (WGS-84) — Moss–Horten core area ──────────────────────────
LAT_MIN, LAT_MAX = 59.27, 59.55
LON_MIN, LON_MAX = 10.32, 10.77

# ── Coordinate transform: WGS-84 → Web Mercator (EPSG:3857) ─────────────────────
_tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def to_merc(lon, lat):
    return _tr.transform(lon, lat)

# Pre-compute map extent in EPSG:3857 (with a small margin)
_margin_lon = (LON_MAX - LON_MIN) * 0.02
_margin_lat = (LAT_MAX - LAT_MIN) * 0.02
X_MIN, Y_MIN = to_merc(LON_MIN - _margin_lon, LAT_MIN - _margin_lat)
X_MAX, Y_MAX = to_merc(LON_MAX + _margin_lon, LAT_MAX + _margin_lat)

# ── Parameters ───────────────────────────────────────────────────────────────────
DENSITY_BINS = 300          # resolution of heatmap grid
VELOCITY_GRID = 25          # cells per axis for quiver grid
VELOCITY_MIN_N = 10         # min pings per cell to draw an arrow
COG_BINS = 200              # resolution of COG heatmap
COG_MIN_N = 2              # min pings per cell to show a COG colour
COG_MIN_COHERENCE = 0.4     # resultant-vector length threshold (0=random, 1=all same)
SOG_MOVING = 0.5            # knots — lower threshold for "underway" vessels


# ── Data loading ─────────────────────────────────────────────────────────────────

def load_raw_ais(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")

    cols = ["mmsi", "latitude", "longitude",
            "speed_over_ground", "course_over_ground", "true_heading"]

    print(f"  Loading {len(files)} parquet files from {data_dir.name} ...")
    df = pd.concat(
        [pd.read_parquet(f, columns=cols) for f in files],
        ignore_index=True,
    )
    print(f"  {len(df):,} raw records")

    df = df.rename(columns={
        "speed_over_ground": "sog",
        "course_over_ground": "cog",
    })

    # valid MMSI
    df = df[(df["mmsi"] >= 100_000_000) & (df["mmsi"] <= 999_999_999)]
    # geographic bounds
    df = df[
        df["latitude"].between(LAT_MIN, LAT_MAX) &
        df["longitude"].between(LON_MIN, LON_MAX)
    ]
    # speed bounds
    df = df[df["sog"].between(0.0, 50.0)]
    # Decimal columns from Parquet → float
    for col in ["sog", "cog", "true_heading"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # sentinel values → NaN
    df.loc[df["cog"] >= 360.0, "cog"] = np.nan
    df.loc[df["true_heading"] == 511, "true_heading"] = np.nan

    # Project to Web Mercator
    df["x"], df["y"] = to_merc(df["longitude"].values, df["latitude"].values)

    print(f"  {len(df):,} records after filtering  |  "
          f"{df['mmsi'].nunique():,} unique vessels")
    return df


# ── Plot 1: Position density heatmap ─────────────────────────────────────────────

def plot_density_heatmap(df: pd.DataFrame, label: str, out_dir: Path) -> None:
    x_edges = np.linspace(X_MIN, X_MAX, DENSITY_BINS + 1)
    y_edges = np.linspace(Y_MIN, Y_MAX, DENSITY_BINS + 1)
    h, _, _ = np.histogram2d(df["x"].values, df["y"].values,
                              bins=[x_edges, y_edges])

    h_masked = np.ma.masked_where(h.T == 0, h.T)

    fig, ax = plt.subplots(figsize=(10, 11))
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)

    ctx.add_basemap(ax, crs="EPSG:3857",
                    source=ctx.providers.CartoDB.Positron,
                    zoom="auto")

    img = ax.pcolormesh(
        x_edges, y_edges, h_masked,
        norm=mcolors.LogNorm(vmin=1),
        cmap="plasma",
        alpha=0.85,
        zorder=2,
    )

    cbar = fig.colorbar(img, ax=ax, fraction=0.035, pad=0.02, extend="max")
    cbar.set_label("AIS ping count (log scale)", fontsize=11)

    ax.set_xlabel("Easting (EPSG:3857)", fontsize=10)
    ax.set_ylabel("Northing (EPSG:3857)", fontsize=10)
    ax.set_title(
        f"AIS Position Density — Moss–Horten, {label}\n"
        f"{len(df):,} pings · {df['mmsi'].nunique():,} vessels · {DENSITY_BINS}×{DENSITY_BINS} grid",
        fontsize=13,
    )
    ax.tick_params(labelsize=8)
    ax.set_aspect("equal")

    out = out_dir / "density_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 2: Velocity vector field ────────────────────────────────────────────────

def plot_velocity_field(df: pd.DataFrame, label: str, out_dir: Path) -> None:
    dv = df.loc[(df["sog"] >= SOG_MOVING) & df["cog"].notna()].copy()
    print(f"  Velocity field: {len(dv):,} underway records")

    x_edges = np.linspace(X_MIN, X_MAX, VELOCITY_GRID + 1)
    y_edges = np.linspace(Y_MIN, Y_MAX, VELOCITY_GRID + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    cog_rad = np.radians(dv["cog"].values)
    dv = dv.copy()
    dv["sin_cog"] = np.sin(cog_rad)
    dv["cos_cog"] = np.cos(cog_rad)
    dv["xi"] = np.clip(np.digitize(dv["x"].values, x_edges) - 1, 0, VELOCITY_GRID - 1)
    dv["yi"] = np.clip(np.digitize(dv["y"].values, y_edges) - 1, 0, VELOCITY_GRID - 1)

    grouped = dv.groupby(["xi", "yi"]).agg(
        n=("sog", "count"),
        mean_sog=("sog", "mean"),
        sin_m=("sin_cog", "mean"),
        cos_m=("cos_cog", "mean"),
    )
    grouped = grouped[grouped["n"] >= VELOCITY_MIN_N]

    U_raw = grouped["sin_m"].values
    V_raw = grouped["cos_m"].values
    mag = np.hypot(U_raw, V_raw)
    U_norm = np.where(mag > 0, U_raw / mag, 0.0)
    V_norm = np.where(mag > 0, V_raw / mag, 0.0)

    arrow_x = x_centers[grouped.index.get_level_values("xi")]
    arrow_y = y_centers[grouped.index.get_level_values("yi")]
    speed_color = np.clip(grouped["mean_sog"].values, 0, 18)

    cell_x_m = (X_MAX - X_MIN) / VELOCITY_GRID
    arrow_len_m = cell_x_m * 0.40
    ref_span = max(X_MAX - X_MIN, Y_MAX - Y_MIN)
    shaft_frac = (arrow_len_m * 0.07) / ref_span

    fig, ax = plt.subplots(figsize=(10, 11))
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")

    ctx.add_basemap(ax, crs="EPSG:3857",
                    source=ctx.providers.CartoDB.Positron,
                    zoom="auto")

    q = ax.quiver(
        arrow_x, arrow_y,
        U_norm * arrow_len_m, V_norm * arrow_len_m,
        speed_color,
        cmap="plasma",
        clim=(0, 18),
        scale=1,
        scale_units="xy",
        width=shaft_frac,
        headwidth=4,
        headlength=5,
        headaxislength=4,
        zorder=3,
    )

    cbar = fig.colorbar(q, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean SOG (knots)", fontsize=11)

    ax.set_xlabel("Easting (EPSG:3857)", fontsize=10)
    ax.set_ylabel("Northing (EPSG:3857)", fontsize=10)
    ax.set_title(
        f"Velocity Vector Field — Moss–Horten, {label}\n"
        f"SOG >= {SOG_MOVING} kn · {VELOCITY_GRID}x{VELOCITY_GRID} grid · "
        f"min {VELOCITY_MIN_N} obs/cell · color = mean SOG",
        fontsize=13,
    )
    ax.tick_params(labelsize=8)

    out = out_dir / "velocity_field.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 3: Mean COG heatmap ──────────────────────────────────────────────────────

def plot_cog_heatmap(df: pd.DataFrame, label: str, out_dir: Path) -> None:
    """Each cell coloured by the circular mean of vessel COG.

    Cyclic HSV colormap: N=red, E=yellow-green, S=cyan, W=blue-purple.
    Cells with low directional coherence (vessels mixing directions) are masked.
    """
    dv = df.loc[(df["sog"] >= SOG_MOVING) & df["cog"].notna()].copy()
    print(f"  COG heatmap: {len(dv):,} underway records")

    x_edges = np.linspace(X_MIN, X_MAX, COG_BINS + 1)
    y_edges = np.linspace(Y_MIN, Y_MAX, COG_BINS + 1)

    cog_rad = np.radians(dv["cog"].values)
    dv["sin_cog"] = np.sin(cog_rad)
    dv["cos_cog"] = np.cos(cog_rad)
    dv["xi"] = np.clip(np.digitize(dv["x"].values, x_edges) - 1, 0, COG_BINS - 1)
    dv["yi"] = np.clip(np.digitize(dv["y"].values, y_edges) - 1, 0, COG_BINS - 1)

    grouped = dv.groupby(["xi", "yi"]).agg(
        n=("sog", "count"),
        sin_m=("sin_cog", "mean"),
        cos_m=("cos_cog", "mean"),
    )

    # Directional coherence = resultant-vector length (0=random, 1=all same heading)
    grouped["coherence"] = np.hypot(grouped["sin_m"], grouped["cos_m"])
    grouped = grouped[
        (grouped["n"] >= COG_MIN_N) & (grouped["coherence"] >= COG_MIN_COHERENCE)
    ]

    # Circular mean COG mapped to [0, 360)
    grouped["mean_cog"] = (
        np.degrees(np.arctan2(grouped["sin_m"], grouped["cos_m"])) + 360
    ) % 360

    # Build 2D grid indexed [xi, yi]; pcolormesh wants shape (ny, nx) → transpose
    cog_grid = np.full((COG_BINS, COG_BINS), np.nan)
    xi = grouped.index.get_level_values("xi").values
    yi = grouped.index.get_level_values("yi").values
    cog_grid[xi, yi] = grouped["mean_cog"].values
    cog_grid_plot = np.ma.masked_invalid(cog_grid.T)

    fig, ax = plt.subplots(figsize=(10, 11))
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")

    ctx.add_basemap(ax, crs="EPSG:3857",
                    source=ctx.providers.CartoDB.Positron,
                    zoom="auto")

    img = ax.pcolormesh(
        x_edges, y_edges, cog_grid_plot,
        cmap="hsv",
        vmin=0, vmax=360,
        alpha=0.85,
        zorder=2,
    )

    cbar = fig.colorbar(img, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean COG (degrees)", fontsize=11)
    cbar.set_ticks([0, 90, 180, 270, 360])
    cbar.set_ticklabels(["N 0°", "E 90°", "S 180°", "W 270°", "N 360°"])

    ax.set_xlabel("Easting (EPSG:3857)", fontsize=10)
    ax.set_ylabel("Northing (EPSG:3857)", fontsize=10)
    ax.set_title(
        f"Mean COG — Moss-Horten, {label}\n"
        f"SOG >= {SOG_MOVING} kn · {COG_BINS}x{COG_BINS} grid · "
        f"min {COG_MIN_N} obs · min coherence {COG_MIN_COHERENCE}",
        fontsize=13,
    )
    ax.tick_params(labelsize=8)

    out = out_dir / "cog_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    keys = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())
    unknown = [k for k in keys if k not in DATASETS]
    if unknown:
        print(f"Unknown dataset(s): {unknown}. Available: {list(DATASETS.keys())}")
        sys.exit(1)

    for key in keys:
        ds = DATASETS[key]
        out_dir = OUTPUT_ROOT / key
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {ds['label']} ===")
        df = load_raw_ais(ds["dir"])

        print("  Plot 1: density heatmap ...")
        plot_density_heatmap(df, ds["label"], out_dir)

        print("  Plot 2: velocity vector field ...")
        plot_velocity_field(df, ds["label"], out_dir)

        print("  Plot 3: mean COG heatmap ...")
        plot_cog_heatmap(df, ds["label"], out_dir)

    print(f"\nDone. Plots in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
