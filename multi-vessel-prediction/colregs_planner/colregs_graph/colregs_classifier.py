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
    6. Narrow channels  — Rule 9 overrides the roles (see ColregsClassifier)

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
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Optional

from colregs_planner.vessel.state import VesselState
from colregs_planner.vessel.encounter import EncounterType
from colregs_planner.colregs_graph.channel import Channel
from colregs_planner.colregs_graph.colregs_graph import (
    ColregsGraph, ColregsEdge, ENCOUNTER_ROLES, VesselRole,
)


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


# ---------------------------------------------------------------------------
# Region lookup table
#
# Every rectangle edge in _REGIONS is a breakpoint, so the (psi, beta) plane
# splits into a grid of cells that each lie entirely inside or outside every
# region.  bisect_right on the sorted breakpoints gives the cell index exactly
# (half-open [min, max) matches the region membership rule), and the table
# holds the first-matching scenario per cell, preserving _REGIONS order as the
# tie-break where rectangles overlap.  Index 0 is the cell below all edges.
# ---------------------------------------------------------------------------

_PSI_EDGES:  list[float] = sorted({e for rs in _REGIONS.values() for r in rs for e in r[:2]})
_BETA_EDGES: list[float] = sorted({e for rs in _REGIONS.values() for r in rs for e in r[2:]})


def _region_lookup(psi_deg: float, beta_deg: float) -> Optional[EncounterType]:
    """First scenario in _REGIONS whose rectangles contain (psi, beta), or None."""
    for scenario, rects in _REGIONS.items():
        for psi_min, psi_max, beta_min, beta_max in rects:
            if psi_min <= psi_deg < psi_max and beta_min <= beta_deg < beta_max:
                return scenario
    return None


def _build_table() -> list[list[Optional[EncounterType]]]:
    # Each cell is sampled at its lower-left corner, which lies inside the
    # cell; the out-of-range cells (index 0) map to None.
    return [
        [_region_lookup(p, b) if i and j else None
         for j, b in enumerate([-math.inf] + _BETA_EDGES)]
        for i, p in enumerate([-math.inf] + _PSI_EDGES)
    ]


_TABLE = _build_table()


def _geometric_classify(psi_deg: float, beta_deg: float) -> Optional[EncounterType]:
    """Return EncounterType from vessel_a's perspective, or None."""
    return _TABLE[bisect_right(_PSI_EDGES, psi_deg)][bisect_right(_BETA_EDGES, beta_deg)]


# ---------------------------------------------------------------------------
# Geometry helpers (NED, radians CW from North)
# ---------------------------------------------------------------------------

_TWO_PI = 2 * math.pi
_RAD2DEG = 180.0 / math.pi


def _wrap_deg(angle: float) -> float:
    """Wrap angle [rad] to [-pi, pi) and convert to degrees."""
    return ((angle + math.pi) % _TWO_PI - math.pi) * _RAD2DEG


def _compute_dcpa_tcpa(a: VesselState, b: VesselState) -> tuple[float, float]:
    """
    Closest Point of Approach distance [m] and time [s].

    tcpa < 0 means vessels are already diverging.
    """
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

    A confirmed overtaking encounter is held until the overtaking vessel is
    past the overtaken vessel's beam (COLREGs Rule 13(d): a later change of
    bearing does not end the duty to keep clear, or turn the encounter into a
    crossing, until she is finally past and clear). Until then it is neither
    cleared nor reclassified, unless the vessels are more than d_a apart.
    Without this, an overtaking vessel that turns away hard makes the TCPA
    go negative and loses the encounter mid-overtake.

    Narrow channels (Rule 9): with channels set, a vessel in channel_bound
    (it can navigate safely only within the channel) that is navigating
    along a channel stands on toward a vessel that is not: that vessel gives
    way, whatever the encounter. If it is crossing the channel this is Rule
    9(d), otherwise 9(b) (a small vessel shall not impede). The edge's rule
    records which. Between two channel-bound vessels, or two that are not,
    the steering rules apply as usual. The override is applied to each
    step's edge, so it follows the vessels in and out of the channel;
    the classified encounter and its hysteresis are unchanged.
    """

    dcpa_max:    float = 1500.0
    d_a:         float = 10000.0
    d_u:         float = 150.0
    n_t:         int   = 11
    w_reinforce: float = 1.0
    w_forget:    float = 0.01
    n_clear:     int   = 5
    channels:    tuple[Channel, ...] = ()
    channel_bound: frozenset[int]    = frozenset()   # vessel indices that can navigate only within a channel

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
                if edge is not None and self.channels:
                    edge = self._rule9(edge, states[id_a], states[id_b])
                if edge is not None:
                    graph.add_edge(edge)

        return graph

    def _rule9(self, edge: ColregsEdge, a: VesselState, b: VesselState) -> ColregsEdge:
        """The edge with Rule 9's roles where a channel-bound vessel navigating
        along a channel meets one that is not (see the class docstring)."""
        for ch in self.channels:
            along_a, along_b = ch.navigating_along(a), ch.navigating_along(b)
            bound_a = along_a and edge.vessel_a in self.channel_bound
            bound_b = along_b and edge.vessel_b in self.channel_bound
            if bound_a == bound_b:
                continue
            other_along = along_b if bound_a else along_a
            rule = "9(b)" if other_along else "9(d)"
            stand, give = VesselRole.STAND_ON, VesselRole.GIVE_WAY
            return ColregsEdge(
                vessel_a=edge.vessel_a, vessel_b=edge.vessel_b,
                role_a=stand if bound_a else give, role_b=give if bound_a else stand,
                has_non_responsive=edge.has_non_responsive,
                encounter_type=edge.encounter_type, rule=rule,
            )
        return edge

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

        # Relative position / velocity of b w.r.t. a, and CPA (tcpa < 0 means
        # the vessels are already diverging).
        dn  = b.north - a.north
        de  = b.east  - a.east
        dvn = b.speed * math.cos(b.psi) - a.speed * math.cos(a.psi)
        dve = b.speed * math.sin(b.psi) - a.speed * math.sin(a.psi)
        dist     = math.hypot(dn, de)
        speed_sq = dvn * dvn + dve * dve
        if speed_sq < 1e-6:
            dcpa, tcpa = dist, math.inf
        else:
            tcpa = -(dn * dvn + de * dve) / speed_sq
            dcpa = math.hypot(dn + tcpa * dvn, de + tcpa * dve)
        approaching = tcpa > 0
        overtaking = self._overtaking_in_progress(a, b, dn, de, dist, ps)

        # --- Hysteresis: clear active scenario if vessels are separating,
        #     beyond the discard distance, AND the backward-projected CPA also
        #     indicates they have truly parted (dcpa > d_u).  The dcpa check
        #     prevents premature clearing when ego's manoeuvre makes the
        #     current-velocity TCPA go negative mid-encounter while the vessels
        #     haven't yet physically separated beyond d_u.
        if ps.active_scenario is not None:
            if dist > self.d_u and not approaching and dcpa > self.d_u and not overtaking:
                ps.clear_count += 1
                if ps.clear_count >= self.n_clear:
                    ps.active_scenario  = None
                    ps.hysteresis_score = 0.0
                    ps.clear_count      = 0
            else:
                ps.clear_count = 0

        # --- Overtaking: hold the encounter as it is until the overtaking
        #     vessel is past (Rule 13(d)); bearing changes do not reclassify it.
        if overtaking:
            return self._active_edge(id_a, id_b, ps)

        # --- DCPA gate: suppress new encounter detection when projected CPA is
        #     very large, but always preserve the already-active edge so the
        #     starboard constraint is not transiently dropped.
        if dcpa > self.dcpa_max:
            return self._active_edge(id_a, id_b, ps)

        # --- Range / approach gate: only classify pairs within d_a and closing
        if dist > self.d_a or not approaching:
            return self._active_edge(id_a, id_b, ps)

        # --- Temporally aware geometric classification
        scenario = self._temporal_classify(a.psi, b.psi, dn, de, dvn, dve, tcpa)
        if scenario is None:
            return self._active_edge(id_a, id_b, ps)

        # --- Hysteresis update
        self._update_hysteresis(ps, scenario)
        return self._active_edge(id_a, id_b, ps)

    def _overtaking_in_progress(
        self, a: VesselState, b: VesselState, dn: float, de: float, dist: float,
        ps: _PairState,
    ) -> bool:
        """True while a confirmed overtaking encounter is not yet past: the
        overtaking vessel is still abaft the overtaken vessel's beam, and the
        vessels are within d_a. (dn, de) is b's position relative to a."""
        if dist > self.d_a or ps.hysteresis_score < self.w_reinforce:
            return False
        if ps.active_scenario == EncounterType.OVERTAKING_AHEAD:      # a overtakes b
            along = -(dn * math.cos(b.psi) + de * math.sin(b.psi))
        elif ps.active_scenario == EncounterType.OVERTAKING_ASTERN:   # b overtakes a
            along = dn * math.cos(a.psi) + de * math.sin(a.psi)
        else:
            return False
        return along < 0.0

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
        psi_a: float, psi_b: float,
        dn: float, de: float,
        dvn: float, dve: float,
        tcpa: float,
    ) -> Optional[EncounterType]:
        """
        Linearly project both vessels to CPA, classify at n_t equally spaced
        steps, and return the priority-weighted mode scenario label.

        (dn, de) and (dvn, dve) are b's position and velocity relative to a.
        Under constant-velocity projection the relative course psi is fixed,
        so only the relative bearing beta(t) of dn + t*dvn, de + t*dve varies.
        The DCPA is also invariant along the projection (same relative line),
        so the caller's DCPA gate covers every projected step.
        """
        row   = _TABLE[bisect_right(_PSI_EDGES, _wrap_deg(psi_b - psi_a))]
        beta0 = _wrap_deg(math.atan2(de, dn) - psi_a)

        if not math.isfinite(tcpa) or tcpa <= 0:
            return row[bisect_right(_BETA_EDGES, beta0)]

        n = self.n_t - 1
        votes: dict[EncounterType, int] = {}

        for k in range(self.n_t):
            t = tcpa * k / n
            beta  = _wrap_deg(math.atan2(de + t * dve, dn + t * dvn) - psi_a)
            label = row[bisect_right(_BETA_EDGES, beta)]
            if label is not None:
                votes[label] = votes.get(label, 0) + 1

        if not votes:
            return row[bisect_right(_BETA_EDGES, beta0)]

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
