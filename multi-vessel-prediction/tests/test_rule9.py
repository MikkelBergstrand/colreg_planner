"""Narrow channels (COLREGs Rule 9): Channel geometry and the classifier's
role override."""

from __future__ import annotations

import contextlib
import io
import math
import sys
from pathlib import Path

import numpy as np

from colregs_planner.colregs_graph.channel import Channel
from colregs_planner.colregs_graph.colregs_classifier import ColregsClassifier
from colregs_planner.colregs_graph.colregs_graph import VesselRole, apply_responsiveness
from colregs_planner.scenario_config import load_config
from colregs_planner.vessel.encounter import EncounterType
from colregs_planner.vessel.state import Intent, VesselState

REPO = Path(__file__).resolve().parents[1]
FERRY = REPO / "scenarios/simulation/milliampere/canal_ferry_crossing.yaml"
CHANNEL = Channel(centreline=[[-100.0, 40.0], [300.0, 40.0]], width=40.0)

# Ferry (0) heading east toward the channel; a vessel (1) coming down it from
# the ferry's port side: a Rule 15 crossing in which the ferry stands on.
FERRY_STATE = VesselState(0.0, 0.0, math.pi / 2, 1.0)
DOWN_CHANNEL = VesselState(60.0, 40.0, math.pi, 1.5)


def _roles(clf, a=FERRY_STATE, b=DOWN_CHANNEL):
    graph = clf.update({0: a, 1: b})
    (edge,) = graph.edges
    return edge


def test_channel_geometry():
    assert CHANNEL.contains(VesselState(0.0, 25.0, 0.0, 1.0))
    assert not CHANNEL.contains(VesselState(0.0, 15.0, 0.0, 1.0))
    assert CHANNEL.navigating_along(VesselState(0.0, 40.0, math.radians(20.0), 1.0))
    assert CHANNEL.navigating_along(VesselState(0.0, 40.0, math.radians(170.0), 1.0))   # either way
    assert not CHANNEL.navigating_along(VesselState(0.0, 40.0, math.radians(60.0), 1.0))
    left, right = CHANNEL.edges()
    assert np.allclose(left[:, 1], 20.0) and np.allclose(right[:, 1], 60.0)


def test_rule15_without_a_channel():
    edge = _roles(ColregsClassifier(d_u=30.0))
    assert edge.encounter_type is EncounterType.CROSSING_PORT
    assert (edge.role_a, edge.role_b, edge.rule) == (VesselRole.STAND_ON, VesselRole.GIVE_WAY, None)


def test_rule9d_crossing_vessel_does_not_impede():
    edge = _roles(ColregsClassifier(d_u=30.0, channels=(CHANNEL,), channel_bound=frozenset({1})))
    assert edge.encounter_type is EncounterType.CROSSING_PORT          # the encounter is unchanged
    assert (edge.role_a, edge.role_b, edge.rule) == (VesselRole.GIVE_WAY, VesselRole.STAND_ON, "9(d)")
    # The rule survives the responsiveness post-processing.
    kept = apply_responsiveness(ColregsClassifier(d_u=30.0, channels=(CHANNEL,), channel_bound=frozenset({1}))
                                .update({0: FERRY_STATE, 1: DOWN_CHANNEL}), {0: Intent.RESPONSIVE, 1: Intent.NON_RESPONSIVE})
    assert kept.edges[0].rule == "9(d)"


def test_rule9b_small_vessel_along_the_channel_does_not_impede():
    small = VesselState(0.0, 35.0, 0.0, 1.0)            # heading up the channel
    big = VesselState(80.0, 45.0, math.pi, 1.5)         # coming down it
    plain = _roles(ColregsClassifier(d_u=30.0), small, big)
    assert plain.role_a is VesselRole.HEAD_ON
    edge = _roles(ColregsClassifier(d_u=30.0, channels=(CHANNEL,), channel_bound=frozenset({1})), small, big)
    assert (edge.role_a, edge.role_b, edge.rule) == (VesselRole.GIVE_WAY, VesselRole.STAND_ON, "9(b)")


def test_rule9_needs_a_channel_bound_vessel():
    edge = _roles(ColregsClassifier(d_u=30.0, channels=(CHANNEL,), channel_bound=frozenset()))
    assert edge.rule is None and edge.role_a is VesselRole.STAND_ON


def test_two_channel_bound_vessels_meet_under_the_steering_rules():
    up = VesselState(0.0, 35.0, 0.0, 1.0)
    down = VesselState(80.0, 45.0, math.pi, 1.5)
    edge = _roles(ColregsClassifier(d_u=30.0, channels=(CHANNEL,), channel_bound=frozenset({0, 1})), up, down)
    assert edge.rule is None and (edge.role_a, edge.role_b) == (VesselRole.HEAD_ON, VesselRole.HEAD_ON)


def test_rule9_applies_only_while_the_bound_vessel_follows_the_channel():
    outside = VesselState(60.0, 70.0, math.pi, 1.5)      # same course, east of the channel
    edge = _roles(ColregsClassifier(d_u=30.0, channels=(CHANNEL,), channel_bound=frozenset({1})),
                  FERRY_STATE, outside)
    assert edge.rule is None


def test_scenario_loads_channels_into_the_classifier():
    config, *_ = load_config(FERRY)
    assert len(config.classifier["channels"]) == 1
    assert config.classifier["channel_bound"] == frozenset({1})
    ColregsClassifier(**config.classifier)


def test_ferry_waits_for_the_channel_vessel():
    sys.path.insert(0, str(REPO / "scripts"))
    import run_simulation as rs

    config, vcfgs, _, pred = load_config(FERRY)
    with contextlib.redirect_stdout(io.StringIO()):
        traj, _, graphs, *_ = rs.simulate_scenario(config, vcfgs, pred.use_responding_filter,
                                                   pred.filter_rollout_model or pred.rollout_model,
                                                   pred_cfg=pred)
    assert graphs[0].edges[0].rule == "9(d)"
    tgt = traj["target_1"]
    assert max(abs(s.east - tgt[0].east) for s in tgt) < 2.0             # kept to its track
    ego, d_pass = traj["ego"], vcfgs["ego"]["d_pass"]
    sep = min(math.hypot(a.north - b.north, a.east - b.east) for a, b in zip(ego, tgt))
    assert sep > d_pass
    # The ferry waited: it was still west of the channel's track when the
    # other vessel crossed its line.
    k = next(i for i, s in enumerate(tgt) if s.north <= 0.0)
    assert ego[k].east < tgt[k].east - d_pass
