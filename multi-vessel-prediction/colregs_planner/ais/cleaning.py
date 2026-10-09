"""colregs_planner/ais/cleaning.py

Raw Kystverket AIS parquet -> cleaned, uniformly-resampled per-voyage tracks.
Shared between scripts/process_raw_ais.py (the encounter-extraction pipeline)
and colregs_planner/nominal/indexer.py (the nominal track-library builder) so both stay
in sync without one importing the other as a script.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

_COG_UNAVAILABLE = 360.0
_HDG_UNAVAILABLE = 511
_ROT_UNAVAILABLE = -128
# Default area (lat_min, lat_max, lon_min, lon_max): the Oslofjord study area.
DEFAULT_BOUNDS: tuple[float, float, float, float] = (59.0, 59.95, 10.2, 10.85)


# ---------------------------------------------------------------------------
# Load & clean one daily parquet file
# ---------------------------------------------------------------------------

def load_and_clean(
    path: Path,
    bounds: tuple[float, float, float, float] = DEFAULT_BOUNDS,
) -> pd.DataFrame:
    """Load one daily parquet file, keeping valid rows inside *bounds*
    (lat_min, lat_max, lon_min, lon_max) [deg]."""
    lat_min, lat_max, lon_min, lon_max = bounds
    df = pd.read_parquet(path)
    df["sog"] = pd.to_numeric(df["speed_over_ground"], errors="coerce")
    df["cog"] = pd.to_numeric(df["course_over_ground"], errors="coerce")
    df.drop(columns=["speed_over_ground", "course_over_ground", "geometry"],
            errors="ignore", inplace=True)
    df.rename(columns={"date_time_utc": "timestamp"}, inplace=True)
    df.loc[df["cog"] >= _COG_UNAVAILABLE, "cog"] = float("nan")
    df.loc[df["true_heading"] == _HDG_UNAVAILABLE, "true_heading"] = float("nan")
    df.loc[df["rate_of_turn"] == _ROT_UNAVAILABLE, "rate_of_turn"] = float("nan")
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    mmsi = df["mmsi"].astype(int)
    df = df[(mmsi >= 100_000_000) & (mmsi <= 999_999_999)].copy()
    df = df[df["latitude"].between(lat_min, lat_max) &
            df["longitude"].between(lon_min, lon_max)].copy()
    df = df[df["sog"].between(0.0, 50.0)].copy()
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset=["mmsi", "timestamp"], keep="first", inplace=True)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Circular COG interpolation
# ---------------------------------------------------------------------------

def _interp_circular_deg(t_new: np.ndarray, t_old: np.ndarray,
                          angles_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cos_v = np.cos(np.radians(angles_deg))
    sin_v = np.sin(np.radians(angles_deg))
    ci = np.interp(t_new, t_old, cos_v)
    si = np.interp(t_new, t_old, sin_v)
    norms = np.hypot(ci, si)
    norms[norms < 1e-9] = 1.0
    return ci / norms, si / norms


def _fill_nan_nearest(vals: np.ndarray) -> np.ndarray:
    """Forward/backward fill NaN using nearest valid value."""
    nans = np.isnan(vals)
    if not nans.any() or nans.all():
        return vals
    idx = np.arange(len(vals))
    valid_idx = idx[~nans]
    fill_pos = np.searchsorted(valid_idx, idx[nans])
    fill_pos = np.clip(fill_pos, 0, len(valid_idx) - 1)
    vals = vals.copy()
    vals[nans] = vals[valid_idx[fill_pos]]
    return vals


# ---------------------------------------------------------------------------
# Trajectory segmentation and resampling
# ---------------------------------------------------------------------------

def resample_track(
    track: pd.DataFrame,
    dt: float,
    gap_threshold: float,
    min_segment_obs: int,
) -> list[pd.DataFrame]:
    track = track.sort_values("timestamp").reset_index(drop=True)
    ts_unix = (track["timestamp"] - pd.Timestamp("1970-01-01")).dt.total_seconds().values
    diffs = np.diff(ts_unix)
    split_at = list(np.where(diffs > gap_threshold)[0] + 1)
    seg_starts = [0] + split_at
    seg_ends = split_at + [len(track)]
    mmsi_val = int(track["mmsi"].iloc[0])
    result: list[pd.DataFrame] = []

    for s0, s1 in zip(seg_starts, seg_ends):
        seg = track.iloc[s0:s1].reset_index(drop=True)
        if len(seg) < min_segment_obs:
            continue
        ts_old = ts_unix[s0:s1]
        span = ts_old[-1] - ts_old[0]
        if span < dt:
            continue
        t_grid = np.arange(ts_old[0], ts_old[-1] + 1e-9, dt)
        out: dict = {}
        out["timestamp"] = pd.to_datetime(t_grid, unit="s")
        out["mmsi"] = mmsi_val

        for col in ["latitude", "longitude", "sog"]:
            vals = seg[col].to_numpy(dtype=float, na_value=float("nan"))
            vals = _fill_nan_nearest(vals)
            out[col] = np.interp(t_grid, ts_old, vals)

        # COG circular interpolation
        cog_vals = seg["cog"].to_numpy(dtype=float, na_value=float("nan"))
        if np.isnan(cog_vals).all():
            cog_vals = np.zeros(len(seg))
        else:
            cog_vals = _fill_nan_nearest(cog_vals)
        ci, si = _interp_circular_deg(t_grid, ts_old, cog_vals)
        out["cos_hdg"] = ci
        out["sin_hdg"] = si
        out["cog"] = np.degrees(np.arctan2(si, ci)) % 360

        hdg_rad = np.arctan2(si, ci)
        dh = np.diff(hdg_rad)
        dh = (dh + math.pi) % (2 * math.pi) - math.pi
        out["heading_rate"] = np.concatenate([[0.0], dh / dt])

        if "true_heading" in seg.columns:
            th = seg["true_heading"].to_numpy(dtype=float, na_value=float("nan"))
            if not np.isnan(th).all():
                th = _fill_nan_nearest(th)
                th_ci, th_si = _interp_circular_deg(t_grid, ts_old, th)
                out["true_heading"] = np.degrees(np.arctan2(th_si, th_ci)) % 360
            else:
                out["true_heading"] = np.full(len(t_grid), float("nan"))

        for col in ["status", "maneuvre", "msg_type"]:
            if col in seg.columns:
                out[col] = np.interp(
                    t_grid, ts_old,
                    seg[col].to_numpy(dtype=float, na_value=0.0)
                )
        if "rate_of_turn" in seg.columns:
            vals = seg["rate_of_turn"].to_numpy(dtype=float, na_value=float("nan"))
            vals = _fill_nan_nearest(vals) if not np.isnan(vals).all() else np.zeros_like(vals)
            out["rate_of_turn"] = np.interp(t_grid, ts_old, vals)
        for col in ["data_source", "ais_class"]:
            if col in seg.columns:
                out[col] = seg[col].iloc[0]

        original_ts = set(np.round(ts_old / dt).astype(int))
        grid_r = np.round(t_grid / dt).astype(int)
        out["interpolated"] = ~np.isin(grid_r, list(original_ts))
        result.append(pd.DataFrame(out))

    return result
