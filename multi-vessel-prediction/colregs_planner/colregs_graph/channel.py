"""colregs_graph/channel.py

Narrow channels and fairways (COLREGs Rule 9) as a centreline polyline and a
width, in the local NED frame.

A vessel is in the channel when it is within width / 2 of the centreline,
and navigating along it when, in addition, its course is within along_deg
of the centreline's direction either way. Anything else in the channel, or
heading into it, is crossing.

Typical usage
-------------
    ch = Channel(centreline=[[-100, 30], [300, 30]], width=40.0)
    ch.navigating_along(state)        # True for a vessel following the channel
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from colregs_planner.vessel.state import VesselState


@dataclass(eq=False)
class Channel:
    centreline: list[list[float]]     # [north, east] vertices [m]
    width:      float                 # [m]
    along_deg:  float = 30.0          # course within this of the centreline counts as along it
    _pts:       np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._pts = np.asarray(self.centreline, dtype=float)
        if len(self._pts) < 2:
            raise ValueError("a channel centreline needs at least two vertices")

    def _nearest(self, north: float, east: float) -> tuple[float, float]:
        """(distance to the centreline [m], direction [rad] of its nearest segment)."""
        p = np.array([north, east])
        a, b = self._pts[:-1], self._pts[1:]
        ab = b - a
        s = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.einsum("ij,ij->i", ab, ab), 0.0, 1.0)
        d = np.linalg.norm(p - (a + s[:, None] * ab), axis=1)
        k = int(np.argmin(d))
        return float(d[k]), math.atan2(ab[k, 1], ab[k, 0])

    def edges(self) -> list[np.ndarray]:
        """The channel's two limits, (k, 2) [north, east] each: the centreline
        offset by width / 2 to either side (vertices offset along the
        mean normal of their segments)."""
        d = np.diff(self._pts, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        t = np.vstack([d[:1], d[:-1] + d[1:], d[-1:]])
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        normal = np.stack([-t[:, 1], t[:, 0]], axis=1)
        return [self._pts + side * 0.5 * self.width * normal for side in (-1.0, 1.0)]

    def contains(self, state: VesselState) -> bool:
        return self._nearest(state.north, state.east)[0] <= 0.5 * self.width

    def navigating_along(self, state: VesselState) -> bool:
        """In the channel, and on a course within along_deg of its axis (either way)."""
        d, axis = self._nearest(state.north, state.east)
        if d > 0.5 * self.width:
            return False
        off = abs(math.atan2(math.sin(state.psi - axis), math.cos(state.psi - axis)))
        return min(off, math.pi - off) <= math.radians(self.along_deg)
