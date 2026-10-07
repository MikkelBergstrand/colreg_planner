"""
Equivalence tests for the optimized ColregsClassifier.

The reference below is the original (pre-optimization) implementation: an
if-chain over _REGIONS and full per-step state projection with a DCPA
re-check.  The optimized classifier must reproduce its graphs exactly,
including hysteresis behaviour over multi-step sequences.

Run with pytest, or directly:  python -m tests.test_colregs_classifier
"""

from __future__ import annotations

import math
import random

import numpy as np

from src.vessel.state import VesselState
from src.colregs_graph.colregs_classifier import (
    ColregsClassifier, _REGIONS, _PRIORITY, _geometric_classify,
)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def _ref_geometric_classify(psi_deg, beta_deg):
    for scenario, rects in _REGIONS.items():
        for psi_min, psi_max, beta_min, beta_max in rects:
            if psi_min <= psi_deg < psi_max and beta_min <= beta_deg < beta_max:
                return scenario
    return None


def _ref_wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _ref_psi_beta(a, b):
    psi_rad  = _ref_wrap(b.psi - a.psi)
    beta_rad = _ref_wrap(math.atan2(b.east - a.east, b.north - a.north) - a.psi)
    return math.degrees(psi_rad), math.degrees(beta_rad)


def _ref_dcpa_tcpa(a, b):
    dp = np.array([b.north - a.north, b.east - a.east])
    dv = np.array([b.vn - a.vn, b.ve - a.ve])
    speed_sq = float(np.dot(dv, dv))
    if speed_sq < 1e-6:
        return float(np.linalg.norm(dp)), math.inf
    tcpa = -float(np.dot(dp, dv)) / speed_sq
    return float(np.linalg.norm(dp + tcpa * dv)), tcpa


def _ref_project(s, dt):
    return VesselState(north=s.north + s.vn * dt, east=s.east + s.ve * dt,
                       psi=s.psi, speed=s.speed, r=s.r)


class _ReferenceClassifier(ColregsClassifier):

    def _classify_pair(self, id_a, a, id_b, b, ps):
        dcpa, tcpa  = _ref_dcpa_tcpa(a, b)
        dist        = float(np.linalg.norm(a.position - b.position))
        approaching = tcpa > 0
        if ps.active_scenario is not None:
            if dist > self.d_u and not approaching and dcpa > self.d_u:
                ps.clear_count += 1
                if ps.clear_count >= self.n_clear:
                    ps.active_scenario  = None
                    ps.hysteresis_score = 0.0
                    ps.clear_count      = 0
            else:
                ps.clear_count = 0
        if dcpa > self.dcpa_max:
            return self._active_edge(id_a, id_b, ps)
        if dist > self.d_a or not approaching:
            return self._active_edge(id_a, id_b, ps)
        scenario = self._ref_temporal(a, b, tcpa)
        if scenario is None:
            return self._active_edge(id_a, id_b, ps)
        self._update_hysteresis(ps, scenario)
        return self._active_edge(id_a, id_b, ps)

    def _ref_temporal(self, a, b, tcpa):
        if not math.isfinite(tcpa) or tcpa <= 0:
            return _ref_geometric_classify(*_ref_psi_beta(a, b))
        votes = {}
        for k in range(self.n_t):
            t = tcpa * k / (self.n_t - 1)
            a_p, b_p = _ref_project(a, t), _ref_project(b, t)
            dcpa_p, _ = _ref_dcpa_tcpa(a_p, b_p)
            if dcpa_p > self.dcpa_max:
                continue
            label = _ref_geometric_classify(*_ref_psi_beta(a_p, b_p))
            if label is not None:
                votes[label] = votes.get(label, 0) + 1
        if not votes:
            return _ref_geometric_classify(*_ref_psi_beta(a, b))
        max_votes = max(votes.values())
        return min((s for s, v in votes.items() if v == max_votes),
                   key=lambda s: _PRIORITY[s])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edges(graph):
    return [(e.vessel_a, e.vessel_b, e.role_a, e.role_b) for e in graph.edges]


def _random_scene(rng, n):
    # Vessels placed on a ring heading roughly at the centre -> encounter-heavy.
    states = {}
    for i in range(n):
        ang = rng.uniform(-math.pi, math.pi)
        r   = rng.uniform(300.0, 4000.0)
        states[i] = VesselState(
            north=r * math.cos(ang), east=r * math.sin(ang),
            psi=ang + math.pi + rng.gauss(0.0, 0.6),
            speed=rng.uniform(0.0, 8.0),
        )
    return states


def _advance(states, dt, rng):
    return {
        i: VesselState(north=s.north + s.vn * dt, east=s.east + s.ve * dt,
                       psi=s.psi + rng.gauss(0.0, 0.05), speed=s.speed)
        for i, s in states.items()
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_lookup_table_matches_region_rectangles():
    # Dense grid (0.25 deg) hits every region edge exactly, plus random points.
    pts = [x / 4 for x in range(-720, 720)]
    for p in pts:
        for b in pts:
            assert _geometric_classify(p, b) == _ref_geometric_classify(p, b), (p, b)
    rng = random.Random(0)
    for _ in range(100_000):
        p, b = rng.uniform(-180, 180), rng.uniform(-180, 180)
        assert _geometric_classify(p, b) == _ref_geometric_classify(p, b), (p, b)


def test_classifier_matches_reference_over_sequences():
    n_edges = 0
    for episode in range(200):
        rng    = random.Random(episode)
        states = _random_scene(rng, rng.randint(2, 6))
        ref, opt = _ReferenceClassifier(), ColregsClassifier()
        for step in range(60):
            g_ref, g_opt = ref.update(states), opt.update(states)
            assert _edges(g_opt) == _edges(g_ref), (episode, step)
            n_edges += len(g_ref.edges)
            states = _advance(states, 5.0, rng)
    # Guard against a vacuous pass where no encounters are ever detected.
    assert n_edges > 1000


if __name__ == "__main__":
    test_lookup_table_matches_region_rectangles()
    test_classifier_matches_reference_over_sequences()
    print("ok")
