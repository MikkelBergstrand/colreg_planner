"""src/ais/windows.py

Helpers for the encounter windows written by scripts/process_raw_ais.py.
"""

from __future__ import annotations

import math

from src.vessel.state import VesselState

_KNOTS_TO_MS = 0.5144


def window_to_frames(w: dict) -> list[dict[str, VesselState]]:
    """Convert a window to a per-timestep list of {str(mmsi): VesselState}."""
    mmsi_list = sorted(w["vessels"].keys())
    n_steps = len(next(iter(w["vessels"].values())))
    frames: list[dict[str, VesselState]] = []
    for t in range(n_steps):
        frame: dict[str, VesselState] = {}
        for mmsi in mmsi_list:
            df = w["vessels"][mmsi]
            if t >= len(df):
                continue
            row = df.iloc[t]
            frame[str(mmsi)] = VesselState(
                north=float(row["north"]),
                east=float(row["east"]),
                psi=math.atan2(float(row["sin_hdg"]), float(row["cos_hdg"])),
                speed=float(row["sog"]) * _KNOTS_TO_MS,
                r=float(row.get("heading_rate", 0.0)),
            )
        if frame:
            frames.append(frame)
    return frames


def traj_limits(
    frames: list[dict[str, VesselState]], padding: float = 300.0
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Square (east, north) plot limits around every position in frames."""
    all_n = [s.north for f in frames for s in f.values()]
    all_e = [s.east  for f in frames for s in f.values()]
    n_min, n_max = min(all_n) - padding, max(all_n) + padding
    e_min, e_max = min(all_e) - padding, max(all_e) + padding
    nr, er = n_max - n_min, e_max - e_min
    if nr > er:
        d = (nr - er) / 2; e_min -= d; e_max += d
    else:
        d = (er - nr) / 2; n_min -= d; n_max += d
    return (e_min, e_max), (n_min, n_max)
