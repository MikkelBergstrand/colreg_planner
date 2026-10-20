"""seachart/hazard.py

Static hazards (land, shoals) as a signed distance map for planning.

The polygons are rasterised once onto a regular grid in the local NED frame,
and each cell holds its distance to the nearest hazard edge [m]: positive in
open water, negative inside a hazard. distance() interpolates the grid
bilinearly at any number of points at once, so a planner can check whole
batches of tracks (plans, steps, 2) with one call.

Points beyond the grid take the value at its nearest edge: the grid extends
`pad` metres beyond the polygons, so far away that is open water.

Typical usage
-------------
    hazards = HazardMap.from_polygons([bank_east, bank_west], resolution=1.0)
    d = hazards.distance(tracks)          # (P, n) metres to the nearest bank
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from matplotlib.path import Path as MplPath


@dataclass
class HazardMap:
    origin:     tuple[float, float]   # (north, east) of cell [0, 0] [m]
    resolution: float                 # cell size [m]
    sdf:        np.ndarray            # (n_north, n_east) signed distance [m]
    polygons:   list[np.ndarray]      # the source polygons, (k, 2) [north, east]

    @classmethod
    def from_polygons(
        cls,
        polygons:   Sequence[Sequence[Sequence[float]]],
        resolution: float = 1.0,
        pad:        float = 100.0,
    ) -> "HazardMap":
        """Signed distance map of *polygons*, each a sequence of
        [north, east] vertices [m], on a grid covering them plus pad."""
        polys = [np.asarray(p, dtype=float) for p in polygons]
        if not polys:
            raise ValueError("HazardMap needs at least one polygon")
        lo = np.min([p.min(axis=0) for p in polys], axis=0) - pad
        hi = np.max([p.max(axis=0) for p in polys], axis=0) + pad
        shape = np.ceil((hi - lo) / resolution).astype(int) + 1
        nn, ee = np.meshgrid(lo[0] + resolution * np.arange(shape[0]),
                             lo[1] + resolution * np.arange(shape[1]), indexing="ij")
        cells = np.column_stack([nn.ravel(), ee.ravel()])
        inside = np.zeros(len(cells), dtype=bool)
        for p in polys:
            inside |= MplPath(p).contains_points(cells)
        # Exact distance from every cell to the nearest polygon edge. Fine at
        # harbour scale; a full chart would want shapely's STRtree or an EDT.
        dist = np.full(len(cells), np.inf)
        for p in polys:
            a, b = p, np.roll(p, -1, axis=0)
            for k in range(len(p)):
                ab = b[k] - a[k]
                s = np.clip((cells - a[k]) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0)
                dist = np.minimum(dist, np.linalg.norm(cells - (a[k] + s[:, None] * ab), axis=1))
        sdf = np.where(inside, -dist, dist).reshape(shape)
        return cls(origin=(float(lo[0]), float(lo[1])), resolution=resolution,
                   sdf=sdf.astype(np.float32), polygons=polys)

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance [m] to the nearest hazard at points (..., 2)
        [north, east]: positive in open water, negative inside a hazard."""
        pts = np.asarray(points, dtype=float)
        f = (pts - np.asarray(self.origin)) / self.resolution
        hi = np.array(self.sdf.shape) - 1
        f = np.clip(f, 0.0, hi)
        i = np.minimum(f.astype(int), hi - 1)
        t = f - i
        g = self.sdf
        i0, i1 = i[..., 0], i[..., 1]
        t0, t1 = t[..., 0], t[..., 1]
        return ((1 - t0) * ((1 - t1) * g[i0, i1] + t1 * g[i0, i1 + 1])
                + t0 * ((1 - t1) * g[i0 + 1, i1] + t1 * g[i0 + 1, i1 + 1]))
