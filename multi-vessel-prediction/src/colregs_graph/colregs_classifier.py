"""
colregs_classifier.py

Observation-driven COLREGs scenario classifier.

Produces a ColregsGraph from a dict of observed VesselStates, using the
Temporally Aware Geometric COLREG Scenario Classification (TAG-CSC) approach
of Gleeson et al. (2024), adapted for the symmetric multi-vessel graph
representation used in this project.

Classification pipeline (per unordered pair {a, b}):
    1. DCPA gate        — pairs with DCPA > dcpa_max get no edge
    2. Geometric lookup — (psi, beta) region table → EncounterType
    3. Temporal vote    — forward-project to CPA, take priority-weighted mode
    4. Hysteresis       — maintain active scenario until vessels are d_u apart
    5. Role assignment  — map EncounterType to (role_a, role_b) via ENCOUNTER_ROLES

Coordinate convention: NED, heading in radians CW from North.

Usage
-----
    classifier = ColregsClassifier()
    while True:
        states: dict[int, VesselState] = get_observations()
        graph: ColregsGraph = classifier.update(states)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.vessel.state import VesselState
from src.vessel.encounter import EncounterType
from src.colregs_graph.colregs_graph import ColregsGraph, ColregsEdge, ENCOUNTER_ROLES


# ---------------------------------------------------------------------------
# Geometric region table  (Gleeson et al. 2024, Table 1)
#
# Regions are defined in terms of:
#   psi  = theta_TS - theta_OS   (relative course, degrees)
#   beta = bearing_to_TS - theta_OS  (relative bearing, degrees)
#
# Each entry is a list of (psi_min, psi_max, beta_min, beta_max) rectangles.
# Membership: psi_min <= psi < psi_max  AND  beta_min <= beta < beta_max.
# The scenario label is always from vessel_a's perspective (a = OS, b = TS).
# ---------------------------------------------------------------------------

_REGIONS: dict[EncounterType, list[tuple[float, float, float, float]]] = {
    EncounterType.HEAD_ON: [
        ( 135,  180, -45,  45),
        (-180, -135, -45,  45),
    ],
    EncounterType.CROSSING_STARBOARD: [   # vessel_a gives way
        (-180,  -64,  45,  180),
        ( -64,    0,  67.5, 112.5),
        (-135,  -64, -30,   45),
    ],
    EncounterType.CROSSING_PORT: [        # vessel_a stands on
        (  66,  180, -180, -45),
        (   0,   66, -112.5, -67.5),
        (  66,  135,  -45,  30),
    ],
    EncounterType.OVERTAKING_AHEAD: [     # vessel_a is overtaking (gives way)
        (-65,  66, -67.5,  67.5),
    ],
    EncounterType.OVERTAKING_ASTERN: [    # vessel_a is being overtaken (stands on)
        (-65,  66, -180, -112.5),
        (-65,  66,  112.5, 180),
    ],
}

# Priority for tie-breaking in temporal vote (lower = higher priority).
# Give-way scenarios are higher priority: failing to give way is more dangerous.
# (Gleeson et al. 2024, Table 2)
_PRIORITY: dict[EncounterType, int] = {
    EncounterType.CROSSING_STARBOARD: 1,
    EncounterType.OVERTAKING_AHEAD:   2,
    EncounterType.HEAD_ON:            3,
    EncounterType.CROSSING_PORT:      4,
    EncounterType.OVERTAKING_ASTERN:  5,
}


def _in_region(
    psi_deg: float,
    beta_deg: float,
    rects: list[tuple[float, float, float, float]],
) -> bool:
    for psi_min, psi_max, beta_min, beta_max in rects:
        if psi_min <= psi_deg < psi_max and beta_min <= beta_deg < beta_max:
            return True
    return False


def _geometric_classify_rects(psi_deg: float, beta_deg: float) -> Optional[EncounterType]:
    """Reference implementation: scan the region rectangles in _REGIONS order."""
    for scenario, rects in _REGIONS.items():
        if _in_region(psi_deg, beta_deg, rects):
            return scenario
    return None


# Precomputed (psi, beta) lookup grid. Every boundary in _REGIONS is a multiple
# of _GRID_RES, so each cell lies entirely inside one region and the lookup is
# exact (up to float rounding of the index right at a boundary).
_GRID_RES = 0.5                       # [deg]
_GRID_SCALE = 1.0 / _GRID_RES
_GRID_N = int(360 * _GRID_SCALE)      # cells per axis, covering [-180, 180)

assert all(
    float(v * _GRID_SCALE).is_integer()
    for rects in _REGIONS.values() for rect in rects for v in rect
), "_REGIONS boundaries must lie on the lookup grid"

def _build_grid() -> list[Optional[EncounterType]]:
    # Paint rectangles in reverse _REGIONS order so earlier scenarios win
    # overlaps, matching the first-match semantics of _geometric_classify_rects.
    scenarios = list(_REGIONS)
    codes = np.full((_GRID_N, _GRID_N), -1, dtype=np.int8)
    for code in reversed(range(len(scenarios))):
        for psi_min, psi_max, beta_min, beta_max in _REGIONS[scenarios[code]]:
            codes[
                int((psi_min + 180) * _GRID_SCALE):int((psi_max + 180) * _GRID_SCALE),
                int((beta_min + 180) * _GRID_SCALE):int((beta_max + 180) * _GRID_SCALE),
            ] = code
    lut = scenarios + [None]   # code -1 -> None
    return [lut[c] for c in codes.ravel().tolist()]


_GRID: list[Optional[EncounterType]] = _build_grid()


def _geometric_classify(psi_deg: float, beta_deg: float) -> Optional[EncounterType]:
    """Return EncounterType from vessel_a's perspective, or None."""
    x = (psi_deg + 180.0) * _GRID_SCALE
    y = (beta_deg + 180.0) * _GRID_SCALE
    if not (0.0 <= x < _GRID_N and 0.0 <= y < _GRID_N):   # also rejects NaN
        return None
    return _GRID[int(x) * _GRID_N + int(y)]


# ---------------------------------------------------------------------------
# Geometry helpers (NED, radians CW from North)
# ---------------------------------------------------------------------------

def _wrap(angle: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _bearing_to(a: VesselState, b: VesselState) -> float:
    """Bearing from a to b [rad], CW from North (NED convention)."""
    dn = b.north - a.north
    de = b.east  - a.east
    return math.atan2(de, dn)


def _compute_psi_beta(a: VesselState, b: VesselState) -> tuple[float, float]:
    """
    Relative course (psi) and relative bearing (beta) from a's perspective,
    in degrees. Both wrapped to (-180, 180].

    psi  = heading_b - heading_a   (relative course)
    beta = bearing_to_b - heading_a  (relative bearing)
    """
    psi_rad  = _wrap(b.psi - a.psi)
    beta_rad = _wrap(_bearing_to(a, b) - a.psi)
    return math.degrees(psi_rad), math.degrees(beta_rad)


def _compute_dcpa_tcpa(a: VesselState, b: VesselState) -> tuple[float, float]:
    """
    Closest Point of Approach distance [m] and time [s].

    tcpa < 0 means vessels are already diverging.
    """
    # Plain scalar math: numpy's per-call overhead dominates on 2-vectors.
    dn  = b.north - a.north
    de  = b.east  - a.east
    dvn = float(b.vn - a.vn)
    dve = float(b.ve - a.ve)

    speed_sq = dvn * dvn + dve * dve
    if speed_sq < 1e-6:
        return math.hypot(dn, de), math.inf

    tcpa = -(dn * dvn + de * dve) / speed_sq
    dcpa = math.hypot(dn + tcpa * dvn, de + tcpa * dve)
    return dcpa, tcpa


# ---------------------------------------------------------------------------
# Per-pair hysteresis state
# ---------------------------------------------------------------------------

@dataclass
class _PairState:
    active_scenario:  Optional[EncounterType] = None
    hysteresis_score: float = 0.0   # in [0, 1]; scenario switches when it hits 0
    clear_count:      int   = 0     # consecutive diverging steps; clears when >= n_clear


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

@dataclass
class ColregsClassifier:
    """
    Stateful COLREGs classifier.  Call update() once per timestep.

    Parameters
    ----------
    dcpa_max : float
        DCPA gate threshold [m]. Pairs with DCPA > dcpa_max get no edge.
    d_a : float
        Maximum range at which a new scenario can be detected [m].
    d_u : float
        Range below which an active scenario is discarded once vessels
        are separating [m].
    n_t : int
        Number of time steps for temporal projection (Gleeson §2.1.2).
    w_reinforce : float
        Hysteresis weight added per timestep when scenario is stable.
    w_forget : float
        Hysteresis weight subtracted per timestep when scenario changes.
        Default 0.0 gives latching behaviour (scenario never forgotten
        until distance check clears it).
    n_clear : int
        Number of consecutive diverging+beyond-d_u steps required before
        an active scenario is cleared. Prevents noisy velocity from
        flickering the tcpa sign and resetting the scenario prematurely.
    """

    dcpa_max:    float = 1500.0
    d_a:         float = 10000.0
    d_u:         float = 150.0
    n_t:         int   = 11
    w_reinforce: float = 1.0
    w_forget:    float = 0.01
    n_clear:     int   = 5

    _pair_states: dict[tuple[int, int], _PairState] = field(
        default_factory=dict, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, states: dict[int, VesselState]) -> ColregsGraph:
        """
        Classify all vessel pairs and return a ColregsGraph.

        Parameters
        ----------
        states : dict[int, VesselState]
            Current observed vessel states keyed by vessel index.
        """
        graph = ColregsGraph()
        ids   = sorted(states.keys())

        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1:]:
                key = (id_a, id_b)  # always id_a < id_b (sorted above)
                if key not in self._pair_states:
                    self._pair_states[key] = _PairState()

                edge = self._classify_pair(
                    id_a, states[id_a],
                    id_b, states[id_b],
                    self._pair_states[key],
                )
                if edge is not None:
                    graph.add_edge(edge)

        return graph

    def reset(self) -> None:
        """Clear all hysteresis state (e.g. between episodes)."""
        self._pair_states.clear()

    # ------------------------------------------------------------------
    # Internal: single pair
    # ------------------------------------------------------------------

    def _classify_pair(
        self,
        id_a: int, a: VesselState,
        id_b: int, b: VesselState,
        ps: _PairState,
    ) -> Optional[ColregsEdge]:

        dcpa, tcpa    = _compute_dcpa_tcpa(a, b)
        dist          = float(np.linalg.norm(a.position - b.position))
        approaching   = tcpa > 0

        # --- Hysteresis: clear active scenario if vessels are separating,
        #     beyond the discard distance, AND the backward-projected CPA also
        #     indicates they have truly parted (dcpa > d_u).  The dcpa check
        #     prevents premature clearing when ego's manoeuvre makes the
        #     current-velocity TCPA go negative mid-encounter while the vessels
        #     haven't yet physically separated beyond d_u.
        if ps.active_scenario is not None:
            if dist > self.d_u and not approaching and dcpa > self.d_u:
                ps.clear_count += 1
                if ps.clear_count >= self.n_clear:
                    ps.active_scenario  = None
                    ps.hysteresis_score = 0.0
                    ps.clear_count      = 0
            else:
                ps.clear_count = 0

        # --- DCPA gate: suppress new encounter detection when projected CPA is
        #     very large, but always preserve the already-active edge so the
        #     starboard constraint is not transiently dropped.
        if dcpa > self.dcpa_max:
            return self._active_edge(id_a, id_b, ps)

        # --- Range / approach gate: only classify pairs within d_a and closing
        if dist > self.d_a or not approaching:
            return self._active_edge(id_a, id_b, ps)

        # --- Temporally aware geometric classification
        scenario = self._temporal_classify(a, b, tcpa)
        if scenario is None:
            return self._active_edge(id_a, id_b, ps)

        # --- Hysteresis update
        self._update_hysteresis(ps, scenario)
        return self._active_edge(id_a, id_b, ps)

    def _active_edge(
        self, id_a: int, id_b: int, ps: _PairState
    ) -> Optional[ColregsEdge]:
        if ps.active_scenario is None:
            return None
        role_a, role_b = ENCOUNTER_ROLES[ps.active_scenario]
        return ColregsEdge(
            vessel_a=id_a, vessel_b=id_b,
            role_a=role_a, role_b=role_b,
            encounter_type=ps.active_scenario,
        )

    # ------------------------------------------------------------------
    # Temporal classification  (Gleeson §2.1.2)
    # ------------------------------------------------------------------

    def _temporal_classify(
        self,
        a: VesselState,
        b: VesselState,
        tcpa: float,
    ) -> Optional[EncounterType]:
        """
        Linearly project both vessels to CPA, classify at n_t equally spaced
        steps, and return the priority-weighted mode scenario label.
        """
        if not math.isfinite(tcpa) or tcpa <= 0:
            return _geometric_classify(*_compute_psi_beta(a, b))

        times = [tcpa * k / (self.n_t - 1) for k in range(self.n_t)]
        votes: dict[EncounterType, int] = {}

        # Constant-heading projection: velocities and relative course are
        # invariant over t, so compute them once and project with scalar math
        # (equivalent to projecting each state and calling _compute_psi_beta on
        # the projected pair). The relative track is also invariant, so DCPA is
        # the same at every projected step; the caller has already gated on it.
        a_vn, a_ve = float(a.vn), float(a.ve)
        b_vn, b_ve = float(b.vn), float(b.ve)
        a_psi      = a.psi
        psi_deg    = math.degrees(_wrap(b.psi - a_psi))

        for t in times:
            dn = (b.north + b_vn * t) - (a.north + a_vn * t)
            de = (b.east  + b_ve * t) - (a.east  + a_ve * t)
            beta_deg = math.degrees(_wrap(math.atan2(de, dn) - a_psi))
            label = _geometric_classify(psi_deg, beta_deg)
            if label is not None:
                votes[label] = votes.get(label, 0) + 1

        if not votes:
            return _geometric_classify(*_compute_psi_beta(a, b))

        max_votes  = max(votes.values())
        candidates = [s for s, v in votes.items() if v == max_votes]
        return min(candidates, key=lambda s: _PRIORITY[s])

    # ------------------------------------------------------------------
    # Hysteresis  (Gleeson §2.1.4)
    # ------------------------------------------------------------------

    def _update_hysteresis(self, ps: _PairState, new_scenario: EncounterType) -> None:
        if ps.active_scenario is None:
            ps.active_scenario  = new_scenario
            ps.hysteresis_score = 0.0
            return

        if new_scenario == ps.active_scenario:
            ps.hysteresis_score = min(1.0, ps.hysteresis_score + self.w_reinforce)
        else:
            ps.hysteresis_score = max(0.0, ps.hysteresis_score - self.w_forget)
            if ps.hysteresis_score <= 0.0:
                ps.active_scenario  = new_scenario
                ps.hysteresis_score = 0.0
