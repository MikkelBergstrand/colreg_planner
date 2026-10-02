"""process_raw_ais.py

Pipeline: raw AIS parquet files -> encounter windows pickle.

Reads the daily Kystverket parquet files, cleans and resamples each vessel
trajectory, then detects multi-vessel encounter groups (2+ vessels closing
on each other) and extracts fixed-length windows.

Key parameters
--------------
  --dt 10        Resample to 10 s (native AIS rate is ~9 s median)
  --steps 120    Window length (120 × 10 s = 20 min)
  --min-vessels 2  Only keep windows with this many simultaneous vessels
  --cpa 2000     CPA threshold for at least one pair in the group [m]
  --tcpa 600     TCPA threshold [s]
  --range 5000   Proximity radius for grouping additional vessels [m]

Output format
-------------
Each window is a dict:

    {
        "window_id"        : int,
        "dt"               : float,
        "rollout_start_idx": int,
        "ref_lat"          : float,
        "ref_lon"          : float,
        "vessels": {
            mmsi (int): pd.DataFrame,   # n_steps rows, all AIS fields + NED
        },
        "pairs": {
            (mmsi_a, mmsi_b): {
                "cpa_t" : np.ndarray (n_steps,) [m],
                "tcpa_t": np.ndarray (n_steps,) [s],
                "alpha_t": np.ndarray (n_steps,) [deg],
                "beta_t" : np.ndarray (n_steps,) [deg],
            }, ...
        },
        "meta": {
            "scenario"      : str,
            "cpa_m"         : float,     # CPA at rollout_start_idx (primary pair)
            "enc_cpa_m"     : float,     # CPA at encounter centre (primary pair)
            "tcpa_s"        : float,
            "n_vessels"     : int,
            "primary_pair"  : (int, int),
        },
    }

Vessel DataFrame columns:
    timestamp, mmsi, lat, lon, sog, cog, cos_hdg, sin_hdg, heading_rate,
    true_heading, status, maneuvre, msg_type, rate_of_turn,
    data_source, ais_class, north, east, interpolated

Usage
-----
    python scripts/process_raw_ais.py
    python scripts/process_raw_ais.py --dt 10 --steps 120 --min-vessels 2
    python scripts/process_raw_ais.py --min-vessels 3
    python scripts/process_raw_ais.py --days 3 --out data/test.pkl
    python scripts/process_raw_ais.py --data-dir "data/<dataset label>" --out data/<dataset label>/processed/ais.pkl
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


from src.ais.cleaning import load_and_clean as _load_and_clean
from src.ais.cleaning import resample_track as _resample_track

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "Moss Horten Jan 2026 AIS"
_DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "Moss Horten Feb 2026 AIS" / "processed" / "ais_04.pkl"

_LAT_TO_M = 111320.0


# ---------------------------------------------------------------------------
# Scenario classification
# ---------------------------------------------------------------------------

def _classify_scenario(alpha_deg: float) -> str:
    if math.isnan(alpha_deg):
        return "unknown"
    a = abs(alpha_deg % 360)
    if a > 180:
        a = 360 - a
    if a < 22.5:
        return "head-on"
    if a > 157.5:
        return "overtaking"
    return "crossing"


# ---------------------------------------------------------------------------
# Merged per-MMSI track (numpy arrays for fast interpolation)
# ---------------------------------------------------------------------------

def _build_merged_tracks(
    tracks_by_mmsi: dict[int, list[pd.DataFrame]],
) -> dict[int, dict[str, np.ndarray | str]]:
    """Concatenate all resampled segments per MMSI into sorted numpy arrays."""
    merged: dict[int, dict] = {}
    for mmsi, segs in tracks_by_mmsi.items():
        if not segs:
            continue
        combined = pd.concat(segs, ignore_index=True).sort_values("timestamp")
        combined.drop_duplicates("timestamp", keep="first", inplace=True)
        ts = (combined["timestamp"] - pd.Timestamp("1970-01-01")).dt.total_seconds().values
        m: dict = {"ts": ts}
        for col in ["latitude", "longitude", "sog", "cos_hdg", "sin_hdg",
                    "heading_rate", "cog", "true_heading", "status",
                    "maneuvre", "msg_type", "rate_of_turn"]:
            if col in combined.columns:
                m[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0.0).values
        for col in ["data_source", "ais_class"]:
            if col in combined.columns:
                m[col] = str(combined[col].iloc[0])
        m["interpolated"] = combined["interpolated"].values if "interpolated" in combined.columns \
            else np.zeros(len(ts), dtype=bool)
        merged[mmsi] = m
    return merged


def _interp_at(m: dict, t: float) -> dict | None:
    """Interpolate a merged track at a single unix timestamp t."""
    ts = m["ts"]
    if t < ts[0] or t > ts[-1]:
        return None
    out = {}
    for col in ["latitude", "longitude", "sog", "cos_hdg", "sin_hdg",
                "heading_rate", "cog", "true_heading"]:
        if col in m:
            out[col] = float(np.interp(t, ts, m[col]))
    for col in ["data_source", "ais_class"]:
        if col in m:
            out[col] = m[col]
    return out


# ---------------------------------------------------------------------------
# CPA / TCPA (vectorized)
# ---------------------------------------------------------------------------

def _cpa_tcpa_vec(
    n_a: np.ndarray, e_a: np.ndarray, vn_a: np.ndarray, ve_a: np.ndarray,
    n_b: np.ndarray, e_b: np.ndarray, vn_b: np.ndarray, ve_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dn = n_b - n_a
    de = e_b - e_a
    dvn = vn_b - vn_a
    dve = ve_b - ve_a
    dv2 = dvn * dvn + dve * dve
    valid = dv2 > 1e-6
    tcpa = np.full_like(dv2, float("inf"))
    tcpa[valid] = -(dn[valid] * dvn[valid] + de[valid] * dve[valid]) / dv2[valid]
    t_cl = np.where(np.isfinite(tcpa), tcpa, 0.0)
    cpa = np.hypot(dn + t_cl * dvn, de + t_cl * dve)
    return cpa, tcpa


# ---------------------------------------------------------------------------
# NED conversion
# ---------------------------------------------------------------------------

def _latlon_to_ned(
    lat: np.ndarray, lon: np.ndarray, ref_lat: float, ref_lon: float
) -> tuple[np.ndarray, np.ndarray]:
    cos_lat = math.cos(math.radians(ref_lat))
    return (lat - ref_lat) * _LAT_TO_M, (lon - ref_lon) * _LAT_TO_M * cos_lat


# ---------------------------------------------------------------------------
# Pair-level encounter detection
# ---------------------------------------------------------------------------

def _detect_pair_encounters(
    ma: int, mb: int,
    merged: dict[int, dict],
    dt: float,
    cpa_max: float,
    tcpa_max: float,
    range_max: float,
    min_sog: float,
) -> list[tuple[float, float, float]]:
    """Return list of (center_ts, cpa, tcpa) for distinct encounter events."""
    ta = merged[ma]["ts"]
    tb = merged[mb]["ts"]
    t_start = max(ta[0], tb[0])
    t_end = min(ta[-1], tb[-1])
    if t_end - t_start < dt:
        return []

    t_grid = np.arange(t_start, t_end + 1e-9, dt)
    lat_a = np.interp(t_grid, ta, merged[ma]["latitude"])
    lon_a = np.interp(t_grid, ta, merged[ma]["longitude"])
    lat_b = np.interp(t_grid, tb, merged[mb]["latitude"])
    lon_b = np.interp(t_grid, tb, merged[mb]["longitude"])
    sog_a = np.interp(t_grid, ta, merged[ma]["sog"])
    sog_b = np.interp(t_grid, tb, merged[mb]["sog"])
    cos_a = np.interp(t_grid, ta, merged[ma]["cos_hdg"])
    sin_a = np.interp(t_grid, ta, merged[ma]["sin_hdg"])
    cos_b = np.interp(t_grid, tb, merged[mb]["cos_hdg"])
    sin_b = np.interp(t_grid, tb, merged[mb]["sin_hdg"])

    ref_lat = (lat_a.mean() + lat_b.mean()) / 2.0
    ref_lon = (lon_a.mean() + lon_b.mean()) / 2.0
    n_a, e_a = _latlon_to_ned(lat_a, lon_a, ref_lat, ref_lon)
    n_b, e_b = _latlon_to_ned(lat_b, lon_b, ref_lat, ref_lon)

    dist = np.hypot(n_b - n_a, e_b - e_a)
    moving = (sog_a >= min_sog) & (sog_b >= min_sog)

    sog_ms_a = sog_a * 0.5144
    sog_ms_b = sog_b * 0.5144
    vn_a = cos_a * sog_ms_a; ve_a = sin_a * sog_ms_a
    vn_b = cos_b * sog_ms_b; ve_b = sin_b * sog_ms_b

    cpa, tcpa = _cpa_tcpa_vec(n_a, e_a, vn_a, ve_a, n_b, e_b, vn_b, ve_b)
    candidate = (dist < range_max) & moving & (tcpa > 0) & (tcpa < tcpa_max) & (cpa < cpa_max)

    if not candidate.any():
        return []

    cand_idx = np.where(candidate)[0]
    merge_steps = max(1, int(60.0 / dt))  # merge candidates within 60 s
    clusters: list[list[int]] = []
    cur = [int(cand_idx[0])]
    for idx in cand_idx[1:]:
        if idx - cur[-1] <= merge_steps:
            cur.append(int(idx))
        else:
            clusters.append(cur)
            cur = [int(idx)]
    clusters.append(cur)

    events = []
    for cluster in clusters:
        best = cluster[int(np.argmin(cpa[cluster]))]
        events.append((float(t_grid[best]), float(cpa[best]), float(tcpa[best])))
    return events


# ---------------------------------------------------------------------------
# Multi-vessel group expansion
# ---------------------------------------------------------------------------

def _expand_to_group(
    primary_pair: tuple[int, int],
    center_ts: float,
    merged: dict[int, dict],
    all_mmsi: list[int],
    range_max: float,
    min_sog: float,
) -> frozenset[int]:
    """Given a primary pair encounter at center_ts, add any nearby moving vessels."""
    vessel_set: set[int] = set(primary_pair)

    # Positions of primary pair at center_ts
    positions: dict[int, tuple[float, float]] = {}
    for mmsi in primary_pair:
        p = _interp_at(merged[mmsi], center_ts)
        if p:
            positions[mmsi] = (p["latitude"], p["longitude"])

    if len(positions) < 2:
        return frozenset(primary_pair)

    ref_lat = sum(v[0] for v in positions.values()) / len(positions)
    cos_lat = math.cos(math.radians(ref_lat))

    for mc in all_mmsi:
        if mc in vessel_set:
            continue
        p = _interp_at(merged[mc], center_ts)
        if p is None or p.get("sog", 0.0) < min_sog:
            continue
        # Check if mc is within range_max of ANY vessel already in the group
        for mv in vessel_set:
            if mv not in positions:
                continue
            pv = positions[mv]
            dn = (p["latitude"] - pv[0]) * _LAT_TO_M
            de = (p["longitude"] - pv[1]) * _LAT_TO_M * cos_lat
            if math.hypot(dn, de) < range_max:
                vessel_set.add(mc)
                positions[mc] = (p["latitude"], p["longitude"])
                break  # added; re-check others against this new member too

    return frozenset(vessel_set)


# ---------------------------------------------------------------------------
# Window metrics computation
# ---------------------------------------------------------------------------

def _compute_pair_metrics(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    ref_lat: float,
    ref_lon: float,
) -> dict[str, np.ndarray]:
    n_a, e_a = _latlon_to_ned(df_a["lat"].values, df_a["lon"].values, ref_lat, ref_lon)
    n_b, e_b = _latlon_to_ned(df_b["lat"].values, df_b["lon"].values, ref_lat, ref_lon)
    sog_ms_a = df_a["sog"].values * 0.5144
    sog_ms_b = df_b["sog"].values * 0.5144
    cos_a = df_a["cos_hdg"].values; sin_a = df_a["sin_hdg"].values
    cos_b = df_b["cos_hdg"].values; sin_b = df_b["sin_hdg"].values
    vn_a = cos_a * sog_ms_a; ve_a = sin_a * sog_ms_a
    vn_b = cos_b * sog_ms_b; ve_b = sin_b * sog_ms_b
    cpa_t, tcpa_t = _cpa_tcpa_vec(n_a, e_a, vn_a, ve_a, n_b, e_b, vn_b, ve_b)
    alpha_t = np.degrees(np.arctan2(e_b - e_a, n_b - n_a)) % 360
    psi_a = np.arctan2(sin_a, cos_a)
    psi_b = np.arctan2(sin_b, cos_b)
    beta_t = np.degrees(((psi_b - psi_a) + math.pi) % (2 * math.pi) - math.pi)
    return {"cpa_t": cpa_t, "tcpa_t": tcpa_t, "alpha_t": alpha_t, "beta_t": beta_t}


# ---------------------------------------------------------------------------
# N-vessel window extraction
# ---------------------------------------------------------------------------

def _extract_window_n(
    vessel_set: frozenset[int],
    center_ts: float,
    merged: dict[int, dict],
    dt: float,
    n_steps: int,
    rollout_steps_before_cpa: int,
    window_id: int,
    min_coverage: float = 0.4,
    min_sog: float = 1.5,
) -> dict | None:
    """Extract an n_steps window centred so CPA falls at step (n_steps - rollout_before - 1)."""
    cpa_step = n_steps - rollout_steps_before_cpa - 1
    win_start = center_ts - cpa_step * dt
    win_end = win_start + (n_steps - 1) * dt
    t_grid = np.arange(win_start, win_end + 1e-9, dt)
    if len(t_grid) != n_steps:
        t_grid = t_grid[:n_steps]

    vessel_dfs: dict[int, pd.DataFrame] = {}

    for mmsi in vessel_set:
        m = merged[mmsi]
        ts = m["ts"]
        # Coverage check: how much of the window is within this vessel's track?
        covered = np.sum((t_grid >= ts[0]) & (t_grid <= ts[-1]))
        if covered < n_steps * min_coverage:
            continue  # skip vessel — not enough data in window

        out: dict = {}
        out["timestamp"] = pd.to_datetime(t_grid, unit="s")
        out["mmsi"] = mmsi

        for col in ["latitude", "longitude", "sog", "cos_hdg", "sin_hdg",
                    "heading_rate", "cog", "true_heading", "status",
                    "maneuvre", "msg_type", "rate_of_turn"]:
            if col in m:
                out[col] = np.interp(t_grid, ts, m[col])

        for col in ["data_source", "ais_class"]:
            if col in m:
                out[col] = m[col]

        # Mark interpolated: rows outside the original track range or not near a sample
        original_ts_set = set(np.round(ts / dt).astype(int))
        grid_r = np.round(t_grid / dt).astype(int)
        out["interpolated"] = ~np.isin(grid_r, list(original_ts_set))

        # Rename lat/lon for consistent API
        df = pd.DataFrame(out)
        df.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)

        # Skip vessels not actively navigating: require median SOG over ±10 steps
        # around the encounter centre to exceed min_sog
        if "sog" in out:
            lo = max(0, cpa_step - 10)
            hi = min(len(df), cpa_step + 11)
            median_sog = float(np.median(df["sog"].iloc[lo:hi].values))
            if median_sog < min_sog:
                continue

        vessel_dfs[mmsi] = df

    if len(vessel_dfs) < 2:
        return None

    # NED reference = mean position at rollout_start_idx
    rs = rollout_steps_before_cpa
    lats = [float(df["lat"].iloc[rs]) for df in vessel_dfs.values() if rs < len(df)]
    lons = [float(df["lon"].iloc[rs]) for df in vessel_dfs.values() if rs < len(df)]
    if not lats:
        return None
    ref_lat = float(np.mean(lats))
    ref_lon = float(np.mean(lons))

    # Add NED
    cos_lat = math.cos(math.radians(ref_lat))
    for df in vessel_dfs.values():
        df["north"] = (df["lat"].values - ref_lat) * _LAT_TO_M
        df["east"] = (df["lon"].values - ref_lon) * _LAT_TO_M * cos_lat

    # Compute all pairwise metrics
    mmsis = sorted(vessel_dfs.keys())
    pair_metrics: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for i, ma in enumerate(mmsis):
        for mb in mmsis[i + 1:]:
            pair_metrics[(ma, mb)] = _compute_pair_metrics(
                vessel_dfs[ma], vessel_dfs[mb], ref_lat, ref_lon
            )

    if not pair_metrics:
        return None

    # Primary pair = minimum CPA at the encounter centre step
    primary_pair = min(
        pair_metrics.keys(),
        key=lambda p: float(pair_metrics[p]["cpa_t"][cpa_step])
        if np.isfinite(pair_metrics[p]["cpa_t"][cpa_step]) else float("inf"),
    )
    pm = pair_metrics[primary_pair]
    enc_cpa = float(pm["cpa_t"][cpa_step])
    if math.isnan(enc_cpa) or not math.isfinite(enc_cpa):
        return None

    return {
        "window_id": window_id,
        "dt": dt,
        "rollout_start_idx": rs,
        "ref_lat": ref_lat,
        "ref_lon": ref_lon,
        "vessels": vessel_dfs,
        "pairs": pair_metrics,
        "meta": {
            "scenario": _classify_scenario(float(pm["alpha_t"][rs])),
            "cpa_m": float(pm["cpa_t"][rs]),
            "enc_cpa_m": enc_cpa,
            "tcpa_s": float(pm["tcpa_t"][rs]),
            "n_vessels": len(vessel_dfs),
            "primary_pair": primary_pair,
        },
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    dt: float = 10.0,
    window_s: float = 1200.0,
    context_s: float = 300.0,
    cpa_max: float = 2000.0,
    tcpa_max: float = 600.0,
    range_max: float = 5000.0,
    gap_threshold: float = 600.0,
    min_segment_obs: int = 6,
    min_sog: float = 1.5,
    min_vessels: int = 2,
    max_days: int | None = None,
    dates: list[str] | None = None,
    out_path: Path = _DEFAULT_OUT,
    data_dir: Path = _DATA_DIR,
) -> list[dict]:
    n_steps = max(4, round(window_s / dt))
    rollout_steps_before_cpa = max(1, round(context_s / dt))

    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        print(f"ERROR: No parquet files found in {data_dir}", file=sys.stderr)
        sys.exit(1)
    if dates:
        files = [f for f in files if any(d in f.name for d in dates)]
        if not files:
            print(f"ERROR: No files matching dates {dates}", file=sys.stderr)
            sys.exit(1)
    elif max_days is not None:
        files = files[:max_days]

    print(f"Processing {len(files)} daily AIS files ...")
    print(f"  dt={dt}s  window={window_s/60:.0f}min ({n_steps} steps)"
          f"  context={context_s/60:.0f}min ({rollout_steps_before_cpa} steps)"
          f"  CPA<{cpa_max}m  TCPA<{tcpa_max}s  range<{range_max}m"
          f"  min_vessels={min_vessels}")

    # ------------------------------------------------------------------ #
    # Step 1: load, clean, resample
    # ------------------------------------------------------------------ #
    tracks_by_mmsi: dict[int, list[pd.DataFrame]] = {}
    total_raw = 0
    for i, f in enumerate(files):
        print(f"  [{i + 1}/{len(files)}] {f.name} ...", flush=True)
        df = _load_and_clean(f)
        total_raw += len(df)
        for mmsi, group in df.groupby("mmsi"):
            segs = _resample_track(group, dt=dt, gap_threshold=gap_threshold,
                                   min_segment_obs=min_segment_obs)
            if segs:
                tracks_by_mmsi.setdefault(int(mmsi), []).extend(segs)

    n_tracks = sum(len(v) for v in tracks_by_mmsi.values())
    print(f"\nRaw rows: {total_raw:,}   Unique MMSI: {len(tracks_by_mmsi)}"
          f"   Track segments: {n_tracks}")

    # ------------------------------------------------------------------ #
    # Step 2: merge segments into fast per-MMSI lookup
    # ------------------------------------------------------------------ #
    print("Building merged track lookup ...")
    merged = _build_merged_tracks(tracks_by_mmsi)
    mmsi_list = sorted(merged.keys())
    n_pairs = len(mmsi_list) * (len(mmsi_list) - 1) // 2
    print(f"  {len(mmsi_list)} vessels  {n_pairs} pairs to check")

    # ------------------------------------------------------------------ #
    # Step 3: pairwise encounter detection
    # ------------------------------------------------------------------ #
    print("Detecting pairwise encounter events ...")
    # all_events: list of (mmsi_a, mmsi_b, center_ts, cpa, tcpa)
    all_events: list[tuple[int, int, float, float, float]] = []
    pairs_with_enc = 0
    for i, ma in enumerate(mmsi_list):
        for mb in mmsi_list[i + 1:]:
            events = _detect_pair_encounters(
                ma, mb, merged, dt, cpa_max, tcpa_max, range_max, min_sog
            )
            if events:
                pairs_with_enc += 1
                for ts, cpa, tcpa in events:
                    all_events.append((ma, mb, ts, cpa, tcpa))

    print(f"  Pairs with encounters: {pairs_with_enc}   "
          f"Raw encounter events: {len(all_events)}")

    # ------------------------------------------------------------------ #
    # Step 4: expand each event to multi-vessel group
    # ------------------------------------------------------------------ #
    print("Expanding to multi-vessel groups ...")
    # group_events: list of (vessel_frozenset, center_ts, cpa, tcpa, primary_pair)
    group_events: list[tuple[frozenset, float, float, float, tuple[int,int]]] = []
    seen_keys: set[tuple[frozenset, int]] = set()

    for ma, mb, ts, cpa, tcpa in sorted(all_events, key=lambda x: x[2]):
        vessel_set = _expand_to_group(
            (ma, mb), ts, merged, mmsi_list, range_max, min_sog
        )
        # Dedup: same group within a half-window span (overlapping windows → keep one)
        dedup_key = (vessel_set, int(round(ts / (dt * n_steps // 2))))
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        if len(vessel_set) >= min_vessels:
            group_events.append((vessel_set, ts, cpa, tcpa, (ma, mb)))

    print(f"  Unique encounter groups (after dedup, >= {min_vessels} vessels):"
          f" {len(group_events)}")

    # Vessel count breakdown
    counts: dict[int, int] = {}
    for ev in group_events:
        n = len(ev[0])
        counts[n] = counts.get(n, 0) + 1
    for n, c in sorted(counts.items()):
        print(f"    {n} vessels: {c} groups")

    # ------------------------------------------------------------------ #
    # Step 5: extract windows
    # ------------------------------------------------------------------ #
    print("Extracting windows ...")
    windows: list[dict] = []
    window_id = 0

    for vessel_set, center_ts, cpa, tcpa, _ in group_events:
        w = _extract_window_n(
            vessel_set, center_ts, merged, dt, n_steps,
            rollout_steps_before_cpa, window_id, min_sog=min_sog,
        )
        if w is not None and w["meta"]["enc_cpa_m"] <= cpa_max:
            windows.append(w)
            window_id += 1

    print(f"\nExtracted {len(windows)} valid windows.")
    if windows:
        scenarios: dict[str, int] = {}
        nv_counts: dict[int, int] = {}
        for w in windows:
            s = w["meta"]["scenario"]
            scenarios[s] = scenarios.get(s, 0) + 1
            n = w["meta"]["n_vessels"]
            nv_counts[n] = nv_counts.get(n, 0) + 1
        print("  Scenarios:", "  ".join(f"{s}:{c}" for s, c in sorted(scenarios.items())))
        print("  Vessel counts:", "  ".join(f"{n}v:{c}" for n, c in sorted(nv_counts.items())))
        enc_cpas = [w["meta"]["enc_cpa_m"] for w in windows]
        print(f"  enc_CPA: {min(enc_cpas):.0f} - {max(enc_cpas):.0f} m"
              f"  (mean {sum(enc_cpas)/len(enc_cpas):.0f} m)")

    # ------------------------------------------------------------------ #
    # Step 6: save
    # ------------------------------------------------------------------ #
    print(f"\nSaving to {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(windows, fh)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved {len(windows)} windows ({size_mb:.1f} MB).")
    return windows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process raw AIS parquet files into multi-vessel encounter windows."
    )
    parser.add_argument("--dt", type=float, default=2.0,
                        help="Resampling interval [s] (default 2)")
    parser.add_argument("--window", type=float, default=20.0,
                        help="Total window duration [min] (default 20)")
    parser.add_argument("--context", type=float, default=5.0,
                        help="History context before encounter, also post-CPA tail [min] (default 5)")
    parser.add_argument("--cpa", type=float, default=2000.0,
                        help="CPA threshold [m] (default 2000)")
    parser.add_argument("--tcpa", type=float, default=600.0,
                        help="Max TCPA [s] (default 600)")
    parser.add_argument("--range", type=float, default=5000.0, dest="range_m",
                        help="Proximity radius for grouping [m] (default 5000)")
    parser.add_argument("--min-vessels", type=int, default=2,
                        help="Minimum vessel count per window (default 2)")
    parser.add_argument("--min-sog", type=float, default=1.5,
                        help="Minimum mean SOG [kn] to include a vessel (default 1.5)")
    parser.add_argument("--days", type=int, default=None,
                        help="Process only first N day files (for testing)")
    parser.add_argument("--date", type=str, nargs="+", default=None, dest="dates",
                        help="Process specific date(s), e.g. --date 2025-07-15  or  --date 2025-07-01 2025-07-02")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                        help=f"Output pickle path (default {_DEFAULT_OUT})")
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR,
                        help=f"Directory of daily AIS parquet files (default {_DATA_DIR})")
    args = parser.parse_args()

    run(
        dt=args.dt,
        window_s=args.window * 60.0,
        context_s=args.context * 60.0,
        cpa_max=args.cpa,
        tcpa_max=args.tcpa,
        range_max=args.range_m,
        min_vessels=args.min_vessels,
        min_sog=args.min_sog,
        max_days=args.days,
        dates=args.dates,
        out_path=args.out,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
