"""
Equivalence tests for the optimised hot paths in colregs_classifier.py.

Each optimisation (grid lookup in _geometric_classify, scalar
_compute_dcpa_tcpa, inlined projection in _temporal_classify) is checked
against a straightforward reference implementation kept here, and the full
ColregsClassifier.update() is checked end-to-end against a classifier wired
to those references.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
import pytest

import src.colregs_graph.colregs_classifier as cc
from src.vessel.encounter import EncounterType
from src.vessel.state import VesselState


# ---------------------------------------------------------------------------
# Reference implementations (pre-optimisation versions)
# ---------------------------------------------------------------------------

def _ref_compute_dcpa_tcpa(a: VesselState, b: VesselState) -> tuple[float, float]:
    dp = np.array([b.north - a.north, b.east - a.east])
    dv = np.array([b.vn    - a.vn,    b.ve   - a.ve  ])

    speed_sq = float(np.dot(dv, dv))
    if speed_sq < 1e-6:
        return float(np.linalg.norm(dp)), math.inf

    tcpa = -float(np.dot(dp, dv)) / speed_sq
    dcpa = float(np.linalg.norm(dp + tcpa * dv))
    return dcpa, tcpa


def _ref_project_state(s: VesselState, dt: float) -> VesselState:
    return VesselState(
        north=s.north + s.vn * dt,
        east =s.east  + s.ve * dt,
        psi  =s.psi,
        speed=s.speed,
        r    =s.r,
    )


_ref_geometric_classify = cc._geometric_classify_rects


class _ReferenceClassifier(cc.ColregsClassifier):
    """ColregsClassifier with the original per-step projection and DCPA re-check."""

    def _temporal_classify(
        self, a: VesselState, b: VesselState, tcpa: float,
    ) -> Optional[EncounterType]:
        if not math.isfinite(tcpa) or tcpa <= 0:
            return _ref_geometric_classify(*cc._compute_psi_beta(a, b))

        times = [tcpa * k / (self.n_t - 1) for k in range(self.n_t)]
        votes: dict[EncounterType, int] = {}

        for t in times:
            a_p = _ref_project_state(a, t)
            b_p = _ref_project_state(b, t)
            dcpa_p, _ = _ref_compute_dcpa_tcpa(a_p, b_p)
            if dcpa_p > self.dcpa_max:
                continue
            label = _ref_geometric_classify(*cc._compute_psi_beta(a_p, b_p))
            if label is not None:
                votes[label] = votes.get(label, 0) + 1

        if not votes:
            return _ref_geometric_classify(*cc._compute_psi_beta(a, b))

        max_votes  = max(votes.values())
        candidates = [s for s, v in votes.items() if v == max_votes]
        return min(candidates, key=lambda s: cc._PRIORITY[s])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_state(rng: random.Random) -> VesselState:
    return VesselState(
        north=rng.uniform(-5e3, 5e3),
        east =rng.uniform(-5e3, 5e3),
        psi  =rng.uniform(-7, 7),      # deliberately outside (-pi, pi]
        speed=rng.uniform(0, 12),
    )


def _converging_scene(rng: random.Random, n: int) -> dict[int, VesselState]:
    """n vessels in a 4 km ring, all heading roughly toward the centre."""
    states = {}
    for i in range(n):
        ang = rng.uniform(-math.pi, math.pi)
        r   = rng.uniform(500, 4000)
        states[i] = VesselState(
            north=r * math.cos(ang),
            east =r * math.sin(ang),
            psi  =ang + math.pi + rng.uniform(-0.6, 0.6),
            speed=rng.uniform(3, 10),
        )
    return states


def _step(rng: random.Random, states: dict[int, VesselState], dt: float = 5.0):
    return {
        k: VesselState(
            north=s.north + s.vn * dt,
            east =s.east  + s.ve * dt,
            psi  =s.psi + rng.gauss(0, 0.01),
            speed=s.speed,
        )
        for k, s in states.items()
    }


def _edge_key(graph) -> list[tuple]:
    return sorted(
        (e.vessel_a, e.vessel_b, e.role_a.name, e.role_b.name, e.encounter_type.name)
        for e in graph.edges
    )


# ---------------------------------------------------------------------------
# _geometric_classify: grid lookup vs rectangle scan
# ---------------------------------------------------------------------------

def test_geometric_classify_random_points():
    rng = random.Random(0)
    for _ in range(100_000):
        psi, beta = rng.uniform(-180, 180), rng.uniform(-180, 180)
        assert cc._geometric_classify(psi, beta) == _ref_geometric_classify(psi, beta), (psi, beta)


def test_geometric_classify_region_boundaries():
    # Quarter-degree sweep hits every boundary and every cell centre, and each
    # boundary value is also probed one hair either side of it.
    sweep = [-180 + k * 0.25 for k in range(1441)]
    boundaries = sorted({v for rects in cc._REGIONS.values() for r in rects for v in r})
    probes = [v + eps for v in boundaries for eps in (-1e-9, 0.0, 1e-9)]

    for p in probes:
        for q in sweep:
            assert cc._geometric_classify(p, q) == _ref_geometric_classify(p, q), (p, q)
            assert cc._geometric_classify(q, p) == _ref_geometric_classify(q, p), (q, p)


@pytest.mark.parametrize("psi, beta", [
    (180.0, 0.0), (0.0, 180.0), (-180.0 - 1e-9, 0.0), (0.0, -180.0 - 1e-9),
    (math.nan, 0.0), (0.0, math.nan), (math.inf, 0.0), (0.0, -math.inf),
])
def test_geometric_classify_out_of_range(psi, beta):
    assert cc._geometric_classify(psi, beta) is None
    assert _ref_geometric_classify(psi, beta) is None


# ---------------------------------------------------------------------------
# _compute_dcpa_tcpa: scalar vs numpy
# ---------------------------------------------------------------------------

def test_compute_dcpa_tcpa_random_pairs():
    rng = random.Random(1)
    for _ in range(50_000):
        a, b = _random_state(rng), _random_state(rng)
        dcpa, tcpa = cc._compute_dcpa_tcpa(a, b)
        ref_dcpa, ref_tcpa = _ref_compute_dcpa_tcpa(a, b)
        assert type(dcpa) is float and type(tcpa) is float
        assert dcpa == pytest.approx(ref_dcpa, rel=1e-12, abs=1e-9)
        assert tcpa == pytest.approx(ref_tcpa, rel=1e-12, abs=1e-9)


@pytest.mark.parametrize("a, b", [
    (VesselState(10, 20, 1.0, 5.0), VesselState(300, -40, 1.0, 5.0)),   # parallel
    (VesselState(10, 20, 1.0, 5.0), VesselState(10, 20, 1.0, 5.0)),     # identical
    (VesselState(0, 0, 0.0, 0.0),   VesselState(100, 0, 0.0, 0.0)),     # both stopped
])
def test_compute_dcpa_tcpa_zero_relative_velocity(a, b):
    dcpa, tcpa = cc._compute_dcpa_tcpa(a, b)
    ref_dcpa, ref_tcpa = _ref_compute_dcpa_tcpa(a, b)
    assert tcpa == ref_tcpa == math.inf
    assert dcpa == pytest.approx(ref_dcpa)


# ---------------------------------------------------------------------------
# _temporal_classify: inlined projection vs per-step projected states
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dcpa_max", [1500.0, 300.0])
def test_temporal_classify_matches_reference(dcpa_max):
    rng = random.Random(2)
    new = cc.ColregsClassifier(dcpa_max=dcpa_max)
    ref = _ReferenceClassifier(dcpa_max=dcpa_max)

    checked = 0
    while checked < 5_000:
        a, b = _random_state(rng), _random_state(rng)
        # _classify_pair only calls _temporal_classify for pairs that pass the
        # DCPA gate; the removed per-step re-check relied on that contract.
        if cc._compute_dcpa_tcpa(a, b)[0] > dcpa_max:
            continue
        tcpa = rng.choice([rng.uniform(0.1, 2000), math.inf, -5.0, 0.0])
        assert new._temporal_classify(a, b, tcpa) == ref._temporal_classify(a, b, tcpa)
        checked += 1


def test_temporal_classify_zero_relative_velocity():
    a = VesselState(0, 0, 0.3, 6.0)
    b = VesselState(500, 200, 0.3, 6.0)
    assert (cc.ColregsClassifier()._temporal_classify(a, b, 100.0)
            == _ReferenceClassifier()._temporal_classify(a, b, 100.0))


# ---------------------------------------------------------------------------
# End-to-end: ColregsClassifier.update()
# ---------------------------------------------------------------------------

def test_update_matches_reference(monkeypatch):
    rng = random.Random(3)
    episodes = []
    for _ in range(10):
        states = _converging_scene(rng, 10)
        episode = [states]
        for _ in range(50):
            states = _step(rng, states)
            episode.append(states)
        episodes.append(episode)

    def run(cls) -> list[list[tuple]]:
        out = []
        for episode in episodes:
            classifier = cls()
            out.extend(_edge_key(classifier.update(s)) for s in episode)
        return out

    new_edges = run(cc.ColregsClassifier)

    # _classify_pair resolves these as module globals; point them at the references.
    monkeypatch.setattr(cc, "_compute_dcpa_tcpa", _ref_compute_dcpa_tcpa)
    monkeypatch.setattr(cc, "_geometric_classify", _ref_geometric_classify)
    ref_edges = run(_ReferenceClassifier)

    assert sum(map(len, ref_edges)) > 1000, "scenario should produce many edges"
    assert new_edges == ref_edges
