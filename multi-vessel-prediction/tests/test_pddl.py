"""
Tests for the PDDL problem generation (src/planner_api/pddl.py).

Needs the planning extra: pip install -e ".[planning]"
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("unified_planning")

from src.planner_api import PredictionService
from src.planner_api.manoeuvres import DEFAULT_MANOEUVRES
from src.planner_api.pddl import build_problem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pddl_planning_demo import DT, SCENARIOS, observe  # noqa: E402

DELAYS = (0.0, 30.0, 60.0, 90.0, 120.0)


def _service_at(t_end: float) -> PredictionService:
    service = PredictionService.from_config(
        {"H": 150, "interaction": {"use_responding_filter": True, "top_k_hypotheses": 4,
                                   "agent": {"d_pass": 150.0, "tcpa_threshold": 600.0}}},
        dt=DT, own_id="ego",
    )
    for k in range(int(t_end / DT) + 1):
        service.step(observe(SCENARIOS["give_way"], k * DT), t=k * DT)
    return service


def test_skipped_delays_follow_a_failure_of_every_fact():
    """Late in the encounter some delays are skipped, and only after the
    manoeuvre has failed safe and every window at an earlier delay."""
    problem = build_problem(_service_at(120.0), d_pass=150.0, delays_s=DELAYS)
    d_min = 150.0 / 2
    skipped = 0
    for m in DEFAULT_MANOEUVRES:
        if m.keeps_course_and_speed:
            assert [k for k in problem.evaluations if k.startswith(f"{m.name}@")] == [f"{m.name}@0"]
            continue
        evaluated = [d for d in DELAYS if f"{m.name}@{d:g}" in problem.evaluations]
        assert evaluated == list(DELAYS[:len(evaluated)])          # a prefix, in order
        if len(evaluated) == len(DELAYS):
            continue
        skipped += 1
        worst = [{o: c["min_distance_m"]
                  for o, c in problem.evaluations[f"{m.name}@{d:g}"]["worst_case_cpa"].items()}
                 for d in evaluated]
        assert any(min(w.values(), default=math.inf) < d_min for w in worst)
        for oid in worst[0]:
            assert any(w[oid] < 150.0 for w in worst)
    assert skipped > 0
