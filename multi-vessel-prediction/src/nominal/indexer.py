"""nominal/indexer.py

Offline builder: raw Kystverket AIS parquet files → TrackLibrary.

Processing mirrors scripts/process_raw_ais.py (same cleaning, gap segmentation,
and circular-COG resampling) but skips encounter detection entirely — all voyage
segments are indexed to maximise geographic coverage.  Positions are stored in
absolute UTM-32N rather than local NED so the library is session-independent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.ais.cleaning import load_and_clean, resample_track
from src.nominal.library import LibraryConfig, TrackLibrary


class TrackIndexer:
    """Build a TrackLibrary from raw Kystverket AIS parquet files.

    Parameters
    ----------
    config : LibraryConfig
        Library dimensions (l, H, stride, dt).
    min_sog_knots : float
        Exclude windows containing any step slower than this [kn].
    gap_threshold_s : float
        Split a voyage at gaps longer than this [s].
    min_segment_obs : int
        Minimum raw AIS observations required to keep a segment.
    geo_bounds : (lat_min, lat_max, lon_min, lon_max)
        Geographic filter applied during parquet loading.
    """

    def __init__(
        self,
        config: LibraryConfig,
        min_sog_knots: float = 1.5,
        gap_threshold_s: float = 600.0,
        min_segment_obs: int = 6,
        geo_bounds: tuple[float, float, float, float] = (59.0, 59.95, 10.2, 10.85),
    ) -> None:
        self._config = config
        self._min_sog_ms = min_sog_knots * 0.5144
        self._gap_threshold = gap_threshold_s
        self._min_segment_obs = min_segment_obs
        self._geo_bounds = geo_bounds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_from_parquet(
        self,
        parquet_paths: Iterable[Path],
        verbose: bool = True,
    ) -> TrackLibrary:
        """Process parquet files and return a populated TrackLibrary.

        Steps per file:
          1. Load and clean (same filters as src.ais.cleaning.load_and_clean)
          2. Resample each MMSI to uniform dt (linear lat/lon/sog; circular COG)
          3. Segment at gaps > gap_threshold_s
          4. Convert lat/lon → absolute UTM-32N via pyproj
          5. Slide a window of length l+H+1 with step=stride; exclude windows
             with any speed < min_sog or incomplete coverage
        """
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
        window_len = self._config.l + self._config.H + 1
        stride = self._config.stride

        pos_chunks: list[np.ndarray] = []
        vel_chunks: list[np.ndarray] = []

        paths = list(parquet_paths)
        n_files = len(paths)

        for fi, path in enumerate(paths):
            if verbose:
                print(f"  [{fi + 1}/{n_files}] {Path(path).name} ...", flush=True, end=" ")

            df = load_and_clean(path)
            segs_added = 0

            for mmsi, group in df.groupby("mmsi"):
                segs = resample_track(
                    group,
                    self._config.dt,
                    self._gap_threshold,
                    self._min_segment_obs,
                )
                for seg in segs:
                    pos_m, vel_m = self._seg_to_arrays(seg, transformer)
                    if pos_m is None:
                        continue
                    n = len(pos_m)
                    for start in range(0, n - window_len + 1, stride):
                        end = start + window_len
                        w_vel = vel_m[start:end]
                        speeds = np.linalg.norm(w_vel, axis=1)
                        if np.any(speeds < self._min_sog_ms):
                            continue
                        l = self._config.l
                        pos_chunks.append(pos_m[start : start + l].copy())
                        vel_chunks.append(vel_m[start + l : end].copy())
                        segs_added += 1

            if verbose:
                print(f"{segs_added} windows")

        if not pos_chunks:
            raise ValueError(
                "No valid subtrajectories found. Check parquet paths, geo_bounds, "
                "min_sog, and that the segment length l+H+1 does not exceed typical "
                "voyage durations at the chosen dt."
            )

        pos_match = np.array(pos_chunks, dtype=np.float32)  # (N, l, 2)
        vel_fwd = np.array(vel_chunks, dtype=np.float32)    # (N, H+1, 2)

        if verbose:
            print(f"\nTotal: {len(pos_match):,} subtrajectories indexed")

        return TrackLibrary.build(pos_match, vel_fwd, self._config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seg_to_arrays(
        seg: pd.DataFrame,
        transformer,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Convert a resampled segment DataFrame to UTM position and velocity arrays.

        Returns
        -------
        pos_m : (n, 2) float64 [north_utm, east_utm] in metres, or None
        vel_m : (n, 2) float64 [vn, ve] in m/s, or None
        """
        if "latitude" not in seg.columns or "cos_hdg" not in seg.columns:
            return None, None

        lat = seg["latitude"].values
        lon = seg["longitude"].values
        if len(lat) < 2:
            return None, None

        east_arr, north_arr = transformer.transform(lon, lat)
        pos_m = np.stack([north_arr, east_arr], axis=1).astype(np.float64)

        sog_ms = seg["sog"].values * 0.5144  # knots → m/s
        vn = seg["cos_hdg"].values * sog_ms
        ve = seg["sin_hdg"].values * sog_ms
        vel_m = np.stack([vn, ve], axis=1).astype(np.float64)

        return pos_m, vel_m
