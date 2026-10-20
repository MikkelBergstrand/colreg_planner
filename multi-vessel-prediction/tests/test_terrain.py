"""Terrain: HazardMap pickling and terrain-aware VO (policies/vo_analytic.py)."""

from __future__ import annotations

import contextlib
import io
import math
import pickle
import sys
from pathlib import Path

import numpy as np

from colregs_planner.colregs_graph.colregs_classifier import ColregsClassifier
from colregs_planner.colregs_graph.colregs_graph import VesselRole
from colregs_planner.prediction.vo_rollout import VOAgentConfig, make_vo_policy
from colregs_planner.scenario_config import load_config
from colregs_planner.seachart.hazard import HazardMap
from colregs_planner.vessel.state import VesselState

REPO = Path(__file__).resolve().parents[1]
EAST_BANK = [[-100.0, 6.0], [300.0, 6.0], [300.0, 60.0], [-100.0, 60.0]]


def test_hazard_map_pickles_as_its_polygons():
    hz = HazardMap.from_polygons([EAST_BANK])
    data = pickle.dumps(hz)
    assert len(data) < 1000                          # not the grid
    assert pickle.loads(data) is hz                  # one map per process


def _head_on_pick(hazards):
    """Heading [rad] a north-bound VO picks when head-on to a vessel ahead
    and a little to port: Rule 14 has it alter to starboard, toward EAST_BANK."""
    cfg = VOAgentConfig(nominal_psi=0.0, ref_speed=1.2, d_pass=20.0, los_lookahead=30.0,
                        hazards=hazards)
    own = VesselState(0.0, 0.0, 0.0, 1.2)
    other = VesselState(60.0, -3.0, math.pi, 1.2)
    graph = ColregsClassifier(d_u=30.0).update({0: own, 1: other})
    assert dict(graph.get_roles(0))[1] is VesselRole.HEAD_ON
    pol = make_vo_policy(cfg, colregs_graph=graph, own_vessel_idx=0, other_vessel_indices=[1])
    a = pol.select_action(own, [other])
    return math.atan2(a.ve, a.vn), cfg


def _closest_along(hz, psi, cfg):
    s = np.linspace(0.0, cfg.ref_speed * cfg.terrain_lookahead_s, 200)
    return float(hz.distance(np.stack([s * math.cos(psi), s * math.sin(psi)], axis=1)).min())


def test_vo_alters_toward_the_bank_only_as_far_as_it_allows():
    hz = HazardMap.from_polygons([EAST_BANK])
    blind, cfg = _head_on_pick(None)
    aware, _ = _head_on_pick(hz)
    assert blind > 0.0 and _closest_along(hz, blind, cfg) < cfg.terrain_clearance
    assert 0.0 <= aware < blind
    # Within the arc's sampling (2.5 deg) of the clearance.
    assert _closest_along(hz, aware, cfg) > cfg.terrain_clearance - 0.5


def test_vo_terrain_out_of_reach_changes_nothing():
    far = HazardMap.from_polygons([[[p[0], p[1] + 500.0] for p in EAST_BANK]])
    assert _head_on_pick(far)[0] == _head_on_pick(None)[0]


def test_canal_narrow_head_on_keeps_both_off_their_banks():
    sys.path.insert(0, str(REPO / "scripts"))
    import run_simulation as rs

    config, vcfgs, _, pred = load_config(REPO / "scenarios/simulation/milliampere/canal_narrow_head_on.yaml")
    with contextlib.redirect_stdout(io.StringIO()):
        traj, *_ = rs.simulate_scenario(config, vcfgs, pred.use_responding_filter,
                                        pred.filter_rollout_model or pred.rollout_model, pred_cfg=pred)
    for vid, tr in traj.items():
        shore = config.hazard_map.distance(np.array([[s.north, s.east] for s in tr])).min()
        assert shore > 4.0, f"{vid} came within {shore:.1f} m of a bank"
