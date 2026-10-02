"""nominal/retriever.py

Online nearest-neighbour retrieval: vessel history → preferred velocity sequence.

For each vessel at time k the retriever:
  1. Transforms the recent l-step NED track to absolute UTM.
  2. Queries the KDTree for candidates whose end position is within radius r.
  3. Applies a heading cone pre-filter (drop candidates > heading_cone_deg off).
  4. Computes shape distance (Frobenius) for retained candidates and applies
     exponential weights w_j ∝ exp(−dist_j / β).
  5. When n_hist > 0 uses historical candidates only; falls back to constant
     velocity only when no candidates survive.
  6. Returns the weighted mean v_pref (H+1, 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.nominal.library import TrackLibrary


@dataclass
class NominalResult:
    v_pref: np.ndarray      # (H+1, 2)  weighted-mean [vn, ve] per rollout step [m/s]
    weights: np.ndarray     # (n_retained,)  normalised; CV candidate is the last entry
    n_hist: int             # number of historical candidates (excludes CV)
    dtw_dists: np.ndarray   # (n_retained,)  raw DTW distances [m] for diagnostics


class NominalRetriever:
    """Online preferred-velocity retriever backed by a TrackLibrary.

    Parameters
    ----------
    library : TrackLibrary
        Pre-built library (loaded via TrackLibrary.load).
    beta : float
        DTW temperature [m] (paper's β, Eq. 1).  Smaller → sharper
        distribution over candidates.
    r : float
        KDTree search radius [m], applied to the subtrajectory end position.
    origin_north_utm, origin_east_utm : float
        UTM-32N coordinates of the playback session's NED origin [m].
        Set via set_origin() once the window's ref_lat/ref_lon are known.
    """

    def __init__(
        self,
        library: TrackLibrary,
        beta: float = 100.0,
        r: float = 200.0,
        max_candidates: int = 500,
        heading_cone_deg: float = 60.0,
        origin_north_utm: float = 0.0,
        origin_east_utm: float = 0.0,
    ) -> None:
        self._library = library
        self._beta = beta
        self._r = r
        self._max_candidates = max_candidates
        self._heading_cone_deg = heading_cone_deg
        self._origin_north = origin_north_utm
        self._origin_east = origin_east_utm

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_origin(self, ref_lat: float, ref_lon: float) -> None:
        """Compute and store the UTM-32N origin from a geographic reference point."""
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
        east, north = t.transform(ref_lon, ref_lat)
        self._origin_north = float(north)
        self._origin_east = float(east)

    @property
    def library(self) -> TrackLibrary:
        return self._library

    # ------------------------------------------------------------------
    # Online query
    # ------------------------------------------------------------------

    def query(
        self,
        recent_pos_ned: np.ndarray,    # (l, 2)  [north, east] local NED [m]
        current_velocity: np.ndarray,  # (2,)    [vn, ve] [m/s]
    ) -> NominalResult:
        """Retrieve candidates and compute the preferred-velocity sequence."""
        l = self._library.config.l
        H = self._library.config.H

        n = recent_pos_ned.shape[0]
        assert 2 <= n <= l and recent_pos_ned.shape[1] == 2, (
            f"recent_pos_ned must be (2..{l}, 2), got {recent_pos_ned.shape}"
        )

        pos_utm = self._ned_to_utm(recent_pos_ned)  # (n, 2) absolute UTM
        end_pos = pos_utm[-1]                        # (2,) current position

        # --- Step 1: KDTree k-nearest query (bounded to max_candidates) ---
        cand_idx = self._library.query_candidates(end_pos, self._r, self._max_candidates)

        # --- Step 2: heading cone pre-filter ---
        # Slice candidates to last n timesteps so shape comparison is length-matched.
        n_raw = len(cand_idx)
        if n_raw > 0 and self._heading_cone_deg < 180.0:
            cand_pos = self._library.pos_match[cand_idx].astype(np.float64)[:, -n:, :]  # (K, n, 2)
            mask = self._heading_filter(cand_pos, pos_utm)
            if mask.any():
                cand_idx = cand_idx[mask]
                cand_pos = cand_pos[mask]
            else:
                cand_idx = np.empty(0, dtype=np.intp)
                cand_pos = np.empty((0, n, 2), dtype=np.float64)
        elif n_raw > 0:
            cand_pos = self._library.pos_match[cand_idx].astype(np.float64)[:, -n:, :]  # (K, n, 2)
        else:
            cand_pos = np.empty((0, n, 2), dtype=np.float64)

        n_hist = len(cand_idx)

        # --- Step 3: shape-distance weighting for historical candidates ---
        if n_hist > 0:
            cand_vel = self._library.vel_fwd[cand_idx].astype(np.float64)    # (K, H+1, 2)
            dtw_hist = self._compute_dtw(cand_pos, pos_utm)
        else:
            cand_vel = np.empty((0, H + 1, 2), dtype=np.float64)
            dtw_hist = np.empty(0)

        # --- Step 4: CV fallback (used only when no historical candidates) ---
        cv_vel = self._cv_fallback(current_velocity)

        if n_hist > 0:
            # Historical candidates only — CV excluded so it cannot crowd out
            # lane-following tracks on straight segments.
            all_vel = cand_vel                          # (K, H+1, 2)
            all_dtw = dtw_hist                          # (K,)
        else:
            all_vel = cv_vel[None, :, :]                # (1, H+1, 2)
            all_dtw = np.array([0.0])

        # --- Step 5: normalise weights ---
        log_w = -all_dtw / self._beta
        log_w -= log_w.max()
        weights = np.exp(log_w)
        weights /= weights.sum()

        # --- Step 6: weighted mean v_pref ---
        v_pref = np.einsum("k,khd->hd", weights, all_vel)  # (H+1, 2)

        return NominalResult(
            v_pref=v_pref,
            weights=weights,
            n_hist=n_hist,
            dtw_dists=all_dtw,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ned_to_utm(self, pos_ned: np.ndarray) -> np.ndarray:
        utm = pos_ned.astype(np.float64).copy()
        utm[:, 0] += self._origin_north
        utm[:, 1] += self._origin_east
        return utm

    def _compute_dtw(
        self,
        candidate_pos: np.ndarray,  # (K, l, 2) UTM [m]
        query_pos: np.ndarray,      # (l, 2)    UTM [m]
    ) -> np.ndarray:                # (K,) shape-distance [m]
        # Centre each window on its own mean for position-invariant shape matching,
        # then compute vectorised Frobenius distance — equivalent to aligned
        # Euclidean DTW (no warping needed since all sequences are the same
        # fixed length l).
        q    = query_pos - query_pos.mean(axis=0)              # (l, 2)
        c    = candidate_pos - candidate_pos.mean(axis=1, keepdims=True)  # (K, l, 2)
        diff = c - q                    # (K, l, 2) — numpy broadcasts (l, 2) across K
        return np.sqrt((diff ** 2).sum(axis=(1, 2)))  # (K,)

    def _heading_filter(
        self,
        cand_pos: np.ndarray,   # (K, l, 2) UTM [m]; columns: [north, east]
        query_pos: np.ndarray,  # (l, 2)
    ) -> np.ndarray:            # (K,) bool mask
        """Keep candidates whose net displacement direction is within
        heading_cone_deg of the query's net displacement direction."""
        q_disp = query_pos[-1] - query_pos[0]        # (2,) [dn, de]
        if float(np.linalg.norm(q_disp)) < 1.0:     # nearly stationary — skip filter
            return np.ones(len(cand_pos), dtype=bool)

        q_head   = np.arctan2(q_disp[1], q_disp[0])       # scalar [rad]
        c_disp   = cand_pos[:, -1, :] - cand_pos[:, 0, :] # (K, 2)
        c_norms  = np.linalg.norm(c_disp, axis=1)          # (K,)
        moving   = c_norms >= 1.0

        # For stationary candidates substitute query heading so they fail the
        # angle test and are excluded.
        c_heads  = np.where(moving, np.arctan2(c_disp[:, 1], c_disp[:, 0]), q_head + np.pi)
        angle_diff = np.abs(np.arctan2(
            np.sin(c_heads - q_head),
            np.cos(c_heads - q_head),
        ))
        return moving & (angle_diff <= np.deg2rad(self._heading_cone_deg))

    def _cv_fallback(self, current_velocity: np.ndarray) -> np.ndarray:
        """Return a constant-velocity forward sequence."""
        H = self._library.config.H
        return np.tile(current_velocity.astype(np.float64), (H + 1, 1))

