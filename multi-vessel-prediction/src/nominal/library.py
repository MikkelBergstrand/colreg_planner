"""nominal/library.py

Offline-built track library: KDTree-indexed subtrajectory store.

Layout
------
    pos_match : (N, l, 2)   float32, absolute UTM-32N [m]  — lookback window
    vel_fwd   : (N, H+1, 2) float32, [vn, ve] [m/s]         — forward window
    KDTree    : over pos_match[:, -1, :]  (subtrajectory end ≡ current position)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree


@dataclass
class LibraryConfig:
    l: int       # lookback steps (matched window length)
    H: int       # prediction horizon steps
    stride: int  # indexing stride used at build time
    dt: float    # uniform resample interval [s]

    @property
    def n_fwd(self) -> int:
        return self.l

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LibraryConfig":
        return cls(**d)


class TrackLibrary:
    """Indexed store of historical subtrajectories.

    Construction
    ------------
    Build offline via TrackIndexer.fit_from_parquet(), then call save().
    Load at runtime via TrackLibrary.load().  Full arrays are kept in RAM;
    the KDTree is rebuilt from pos_match[:, -1, :] on load (~1 s for 240 K entries).
    """

    def __init__(
        self,
        pos_match: np.ndarray,   # (N, l, 2)
        vel_fwd: np.ndarray,     # (N, H+1, 2)
        config: LibraryConfig,
    ) -> None:
        assert pos_match.ndim == 3 and pos_match.shape[2] == 2
        assert vel_fwd.ndim == 3 and vel_fwd.shape[2] == 2
        assert pos_match.shape[0] == vel_fwd.shape[0], "N mismatch"
        assert pos_match.shape[1] == config.l, "pos_match length != config.l"
        assert vel_fwd.shape[1] == config.H + 1, "vel_fwd horizon != config.H + 1"

        self.config = config
        # copy=False: a loaded library is already float32 (~2 GB), no second copy
        self._pos_match = pos_match.astype(np.float32, copy=False)
        self._vel_fwd = vel_fwd.astype(np.float32, copy=False)
        self._kdtree = KDTree(self._pos_match[:, -1, :].astype(np.float64))

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        pos_match: np.ndarray,
        vel_fwd: np.ndarray,
        config: LibraryConfig,
    ) -> "TrackLibrary":
        return cls(pos_match, vel_fwd, config)

    @classmethod
    def load(cls, path: Path) -> "TrackLibrary":
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required to load a TrackLibrary.  pip install h5py") from exc
        with h5py.File(Path(path), "r") as f:
            pos_match = f["pos_match"][:]
            vel_fwd = f["vel_fwd"][:]
            config = LibraryConfig.from_dict(json.loads(f.attrs["config"]))
        return cls(pos_match, vel_fwd, config)

    def save(self, path: Path) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required to save a TrackLibrary.  pip install h5py") from exc
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.create_dataset("pos_match", data=self._pos_match, compression="gzip", chunks=True)
            f.create_dataset("vel_fwd", data=self._vel_fwd, compression="gzip", chunks=True)
            f.attrs["config"] = json.dumps(self.config.to_dict())

    # ------------------------------------------------------------------
    # Spatial query
    # ------------------------------------------------------------------

    def query_candidates(
        self,
        end_pos_utm: np.ndarray,  # (2,) absolute UTM [north_m, east_m]
        radius: float,            # [m]
        max_candidates: int = 0,  # 0 = unlimited (slow in dense areas)
    ) -> np.ndarray:
        """Return indices of the nearest entries within *radius* of *end_pos_utm*.

        When *max_candidates* > 0 the KDTree k-nearest query is used so at most
        that many entries are returned, avoiding large memory loads in dense lanes.
        """
        if max_candidates > 0:
            k = min(max_candidates, len(self))
            dists, idxs = self._kdtree.query(
                end_pos_utm.astype(np.float64),
                k=k,
                distance_upper_bound=float(radius),
            )
            dists = np.atleast_1d(dists)
            idxs  = np.atleast_1d(idxs)
            valid = dists < radius
            return idxs[valid].astype(np.intp)
        idxs = self._kdtree.query_ball_point(end_pos_utm.astype(np.float64), radius)
        return np.array(idxs, dtype=np.intp)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._pos_match.shape[0]

    @property
    def pos_match(self) -> np.ndarray:
        return self._pos_match

    @property
    def vel_fwd(self) -> np.ndarray:
        return self._vel_fwd
