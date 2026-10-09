"""
Tests for the nominal track-library builder (colregs_planner/nominal/indexer.py) and the
AIS loading it shares with the encounter pipeline (colregs_planner/ais/cleaning.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from colregs_planner.ais.cleaning import DEFAULT_BOUNDS, load_and_clean
from colregs_planner.nominal.indexer import TrackIndexer
from colregs_planner.nominal.library import LibraryConfig

# Two vessels sailing north at 10 kn, both inside the default area: one in
# its south-west, one in its north-east.
SOUTH_WEST = (59.10, 10.30)
NORTH_EAST = (59.80, 10.75)


def _write_ais(path, starts=(SOUTH_WEST, NORTH_EAST), n=60, dt_s=10.0):
    rows = []
    t0 = pd.Timestamp("2024-06-01 12:00:00")
    for k, (lat0, lon0) in enumerate(starts):
        for i in range(n):
            rows.append({
                "mmsi": 257_000_001 + k,
                "date_time_utc": t0 + pd.Timedelta(seconds=i * dt_s),
                "latitude": lat0 + i * dt_s * 10 * 0.5144 / 111_320.0,
                "longitude": lon0,
                "speed_over_ground": 10.0,
                "course_over_ground": 0.0,
                "true_heading": 0,
                "rate_of_turn": 0,
            })
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_load_and_clean_keeps_only_rows_inside_bounds(tmp_path):
    path = _write_ais(tmp_path / "day.parquet")
    assert load_and_clean(path)["mmsi"].nunique() == 2          # default area: both
    south_west = (59.0, 59.5, 10.2, 10.5)
    df = load_and_clean(path, south_west)
    assert set(df["mmsi"]) == {257_000_001}
    assert df["latitude"].between(59.0, 59.5).all() and df["longitude"].between(10.2, 10.5).all()


def test_track_indexer_applies_geo_bounds(tmp_path):
    path = _write_ais(tmp_path / "day.parquet")
    config = LibraryConfig(l=3, H=3, stride=1, dt=10.0)

    def library(bounds):
        return TrackIndexer(config, geo_bounds=bounds).fit_from_parquet([path], verbose=False)

    both = library(DEFAULT_BOUNDS)
    south_west = library((59.0, 59.5, 10.2, 10.5))
    assert len(south_west) == len(both) // 2
    # Positions are stored in UTM-32N: the south-west vessel is the one further south.
    assert south_west.pos_match[:, :, 0].max() < both.pos_match[:, :, 0].max()
    # An area with no traffic gives no library at all.
    with pytest.raises(ValueError, match="geo_bounds"):
        library((58.0, 58.5, 9.0, 9.5))
