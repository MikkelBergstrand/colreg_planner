"""
Tests for the manoeuvre-search planner (src/planner_api/manoeuvre_search.py),
the search_dp policy, and their wiring into the simulation and the plot.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from src.colregs_graph.colregs_classifier import ColregsClassifier
from src.colregs_graph.colregs_graph import VesselRole
from src.planner_api.manoeuvre_search import (
    DEFAULT_ACTIONS, ManoeuvreSearchPlanner, SearchAction, SearchConfig, interpolate_track,
)
from src.policies.search_dp import SearchDPPolicy
from src.prediction.vo_rollout import rank_hypotheses
from src.scenario_config import build_vo_agent_config, load_config
from src.vessel.state import Intent, VesselState

REPO = Path(__file__).resolve().parents[1]
HEAD_ON = REPO / "scenarios/simulation/milliampere/head_on.yaml"


def _planner(path=HEAD_ON, **cfg):
    config, vcfgs, _, pred = load_config(path)
    agent_configs = {
        vi.vessel_id: build_vo_agent_config(vcfgs[vi.vessel_id], vi, policy_type=pred.policy_type,
                                            rollout_model=pred.rollout_model)
        for vi in config.vessels
    }
    index_map = {vi.vessel_id: i for i, vi in enumerate(config.vessels)}
    planner = ManoeuvreSearchPlanner("ego", agent_configs, index_map,
                                     own_intent=config.vessels[0].intent,
                                     config=SearchConfig(**cfg))
    states = {vi.vessel_id: VesselState(vi.state.north, vi.state.east, vi.state.heading,
                                        vi.state.speed) for vi in config.vessels}
    clf = ColregsClassifier(**config.classifier)
    graph = clf.update({index_map[k]: s for k, s in states.items()})
    return planner, states, graph, config


def _simulate(path, policy, **overrides):
    sys.path.insert(0, str(REPO / "scripts"))
    import run_simulation as rs

    config, vcfgs, _, pred = load_config(path)
    vcfgs["ego"]["policy"] = policy
    vcfgs["ego"].update(overrides)
    log = {}
    with contextlib.redirect_stdout(io.StringIO()):
        out = rs.simulate_scenario(config, vcfgs, pred.use_responding_filter,
                                   pred.filter_rollout_model or pred.rollout_model,
                                   pred_cfg=pred, plan_log=log)
    return out, log, config, vcfgs, pred


# ---------------------------------------------------------------------------
# Hypothesis ranking (factored out of enumerate_and_rollout)
# ---------------------------------------------------------------------------

def test_rank_hypotheses_orders_and_drops_zero():
    ranked = rank_hypotheses(["a", "b"], {"a": 0.9, "b": 0.6}, top_k=4)
    probs = [p for _, p in ranked]
    assert probs == pytest.approx([0.54, 0.36, 0.06, 0.04])
    assert ranked[0][0] == {"a": Intent.RESPONSIVE, "b": Intent.RESPONSIVE}
    assert ranked[1][0] == {"a": Intent.RESPONSIVE, "b": Intent.NON_RESPONSIVE}
    # A pinned vessel (p = 1) never appears non-responsive.
    pinned = rank_hypotheses(["a", "b"], {"a": 1.0, "b": 0.5}, top_k=4)
    assert len(pinned) == 2 and all(a["a"] == Intent.RESPONSIVE for a, _ in pinned)


# ---------------------------------------------------------------------------
# Planner building blocks
# ---------------------------------------------------------------------------

def test_velocity_sequence_edges_then_nominal():
    planner, *_ = _planner(edge_s=10.0, plan_dt=1.0, horizon_s=40.0)
    stbd60 = SearchAction("s", course_offset=60.0)
    slow = SearchAction("slow", speed_factor=0.5)
    seq = planner.velocity_sequence((stbd60, slow))
    nominal = planner._nominal_velocity()
    U, psi = np.linalg.norm(nominal), math.atan2(nominal[1], nominal[0])

    assert seq.shape == (40, 2)
    np.testing.assert_allclose(seq[:10], [[U * math.cos(psi + math.radians(60)),
                                           U * math.sin(psi + math.radians(60))]] * 10)
    np.testing.assert_allclose(seq[10:20], [nominal * 0.5] * 10)
    np.testing.assert_allclose(seq[20:], [nominal] * 20)


def test_search_is_exhaustive_and_ranked():
    planner, states, graph, _ = _planner(horizon_s=40.0)
    result = planner.plan(states, graph, t=0.0)
    seqs = [p.actions for p in result.ranked]
    assert sorted(seqs, key=str) == sorted(itertools.product(DEFAULT_ACTIONS, repeat=2), key=str)
    costs = [p.cost for p in result.ranked]
    assert costs == sorted(costs)
    assert result.best is result.ranked[0]
    assert len(result.own_traj) == 41


@pytest.mark.parametrize("probs", [None, {"target_1": 0.99}, {"target_1": 0.0}])
def test_head_on_alters_to_starboard_now(probs):
    # Rule 14 (both alter to starboard) and Rule 16 (early action): even when
    # the target is sure to give way, holding course on the first edge costs.
    planner, states, graph, _ = _planner()
    best = planner.plan(states, graph, t=0.0, responding_probs=probs).best
    assert best.actions[0].course_offset > 0
    assert not any(a.course_offset < 0 for a in best.actions)


def test_non_responsive_own_ship_only_considers_nominal():
    planner, states, graph, _ = _planner(REPO / "scenarios/simulation/milliampere/multi_crossing.yaml")
    assert planner.own_intent == Intent.NON_RESPONSIVE
    result = planner.plan(states, graph, t=0.0)
    assert len(result.ranked) == 1
    assert all(a.is_nominal for a in result.best.actions)


def test_hypotheses_pin_own_intent_and_renormalise():
    planner, states, graph, _ = _planner(horizon_s=20.0)
    result = planner.plan(states, graph, t=0.0,
                          responding_probs={"ego": 0.2, "target_1": 0.7})
    assert len(result.hypotheses) == 2
    assert all(a["ego"] == Intent.RESPONSIVE for a, _ in result.hypotheses)
    assert sum(p for _, p in result.hypotheses) == pytest.approx(1.0)
    assert [p for _, p in result.hypotheses] == pytest.approx([0.7, 0.3])


def test_commitment_lasts_one_edge():
    planner, *_ = _planner(edge_s=15.0)
    a, b = DEFAULT_ACTIONS[1], DEFAULT_ACTIONS[2]
    planner._active = (a, 10.0)
    assert planner._committed_action(20.0) is a
    assert planner._committed_action(25.0) is None      # edge done: free to switch


def test_interpolate_track():
    traj = [VesselState(0.0, 0.0, math.radians(170), 1.0),
            VesselState(2.0, 4.0, math.radians(-170), 1.0)]
    n, e, psi = interpolate_track(traj, 1.0, 0.5)
    assert (n, e) == pytest.approx((1.0, 2.0))
    assert abs(abs(psi) - math.pi) < 1e-9                 # wraps through 180, not through 0
    assert interpolate_track(traj, 1.0, 5.0)[:2] == (2.0, 4.0)


# ---------------------------------------------------------------------------
# SearchDPPolicy
# ---------------------------------------------------------------------------

def test_policy_replans_on_schedule_and_follows_plan():
    planner, states, graph, _ = _planner(horizon_s=30.0)
    pol = SearchDPPolicy(dt=0.5, planner=planner, replan_s=2.0)
    assert pol.needs_replan()
    pol.set_plan(planner.plan(states, graph, t=pol.time))
    traj = pol.current.result.own_traj

    for k in range(4):
        assert not pol.needs_replan()
        a = pol.select_action(states["ego"], [])
        n0, e0, _ = interpolate_track(traj, 1.0, 0.5 * k)
        n1, e1, psi1 = interpolate_track(traj, 1.0, 0.5 * (k + 1))
        assert (a.d_north, a.d_east, a.psi_ref) == pytest.approx((n1 - n0, e1 - e0, psi1))
    assert pol.needs_replan()                              # 2 s elapsed


def test_planned_velocities_resample_plan_in_force():
    planner, states, graph, _ = _planner(horizon_s=30.0, edge_s=10.0)
    pol = SearchDPPolicy(dt=0.5, planner=planner)
    assert pol.planned_velocities(0.0, 0.5, 4) is None
    pol.set_plan(planner.plan(states, graph, t=0.0))
    seq = pol.current.velocity_seq
    v = pol.planned_velocities(3.0, 0.5, 80)
    assert v.shape == (80, 2)
    np.testing.assert_allclose(v[0], seq[3])
    np.testing.assert_allclose(v[-1], seq[-1])             # beyond the horizon: held


# ---------------------------------------------------------------------------
# Simulation and plot wiring
# ---------------------------------------------------------------------------

def test_search_dp_head_on_passes_clear_and_returns_to_route():
    (traj, *_), log, config, vcfgs, _ = _simulate(HEAD_ON, "search_dp")
    ego, tgt = traj["ego"], traj["target_1"]
    d_min = min(math.hypot(p.north - q.north, p.east - q.east) for p, q in zip(ego, tgt))
    assert d_min >= vcfgs["ego"]["d_pass"]
    assert max(s.east for s in ego) > 10.0                 # altered to starboard (east)
    # Heading back to the route after the encounter (slowly: the turn back
    # costs along-route progress, which the time term charges).
    assert abs(ego[-1].east) < 0.6 * max(abs(s.east) for s in ego)
    assert ego[-1].east < ego[-40].east
    plans = log["ego"].plans
    assert len(plans) == math.ceil(config.t_horizon / 5.0)
    assert [p.t for p in plans] == pytest.approx([5.0 * i for i in range(len(plans))])


def test_search_dp_requires_milliampere():
    with pytest.raises(ValueError, match="search_dp"):
        _simulate(HEAD_ON, "search_dp", sim_model="ct_decoupled_ar")


def test_unknown_search_key_rejected():
    with pytest.raises(ValueError, match="dpeth"):
        _simulate(HEAD_ON, "search_dp", search={"dpeth": 3})


def test_drawn_prediction_follows_committed_plan():
    (traj, _, _, snaps, filt, hist), log, config, vcfgs, pred = _simulate(HEAD_ON, "search_dp")
    import run_simulation as rs

    H = round(pred.horizon / config.dt)
    pol = log["ego"]
    pred_fn = rs._make_pred_fn(
        config.vessels, vcfgs, config.dt, H, config,
        policy_type=pred.policy_type, rollout_model=pred.rollout_model,
        classifier_snapshots=snaps, responding_filters=filt, responding_history=hist,
        top_k=1, reclassify_in_rollout=pred.reclassify_in_rollout,
        fixed_probs=rs._dp_own_intent_probs(config, vcfgs),
        fixed_plans=lambda f: {"ego": pol.planned_velocities(f * config.dt, config.dt, H)},
    )
    k = 2                                                  # mid-turn of the first plan
    drawn = pred_fn({vid: traj[vid][k] for vid in traj}, k)["ego"]
    plan_traj = pol.plans[0].result.own_traj               # same plan, from t = 0
    n, e, _ = interpolate_track(plan_traj, pol.plans[0].result.plan_dt, k * config.dt + 20.0)
    assert math.hypot(drawn[40].north - n, drawn[40].east - e) < 3.0


def test_plan_overlay_marks_each_manoeuvre():
    (traj, *_), log, config, _, _ = _simulate(HEAD_ON, "search_dp")
    import run_simulation as rs

    overlays = rs._plan_overlay_fn(log, config.dt)
    pol = log["ego"]
    assert overlays(0)["ego"].marks[0][2].endswith("(t=0 s)")

    k = 13                                       # 6.5 s: the plan made at t = 5 s is in force
    rec = pol.plans[1]
    ov = overlays(k)["ego"]
    assert len(ov.north) == len(rec.result.own_traj)
    labels = [m[2] for m in ov.marks]
    names = [a.name for a in rec.result.best.actions]
    assert labels == [f"{names[0]} (t=5 s)", *names[1:], "nominal"]
    # The first mark is where the plan starts: ego's position at replan time.
    start = traj["ego"][10]
    assert (ov.marks[0][0], ov.marks[0][1]) == pytest.approx((start.north, start.east))


# ---------------------------------------------------------------------------
# Stopping
# ---------------------------------------------------------------------------

GIVE_WAY_STOP = REPO / "scenarios/simulation/milliampere/give_way_stop.yaml"


def test_fixed_plan_stop_holds_vessel_in_rollout():
    from dataclasses import replace
    from src.prediction.vo_rollout import VOAgentConfig, vo_rollout

    seq = np.vstack([np.zeros((10, 2)), np.tile([1.5, 0.0], (10, 1))])
    cfg = replace(VOAgentConfig(nominal_psi=0.0, ref_speed=1.5, origin_north=0.0,
                                origin_east=0.0), pref_vel_seq=seq)
    tr = vo_rollout({"ego": VesselState(0.0, 0.0, 0.0, 1.5)}, 1.0, 20, {"ego": cfg},
                    fixed_plan_ids={"ego"})["ego"]
    assert [round(float(tr[k].north), 6) for k in (0, 5, 10)] == [0.0, 0.0, 0.0]
    assert tr[15].north == pytest.approx(7.5)
    assert all(s.psi == 0.0 for s in tr)                  # heading kept while stopped


def test_stop_action_zero_velocity_and_costs_time():
    planner, states, graph, _ = _planner(edge_s=15.0, plan_dt=1.0, horizon_s=60.0)
    stop = next(a for a in DEFAULT_ACTIONS if a.name == "stop")
    nominal = next(a for a in DEFAULT_ACTIONS if a.is_nominal)
    seq = planner.velocity_sequence((stop, nominal))
    np.testing.assert_allclose(seq[:15], 0.0)
    result = planner.plan(states, graph, t=0.0)
    by_actions = {p.actions: p for p in result.ranked}
    assert by_actions[(stop, nominal)].terms["time"] == pytest.approx(1.0, abs=0.05)
    assert by_actions[(stop, stop)].terms["time"] == pytest.approx(2.0, abs=0.05)
    assert by_actions[(nominal, nominal)].terms["time"] == pytest.approx(0.0, abs=0.05)


def test_give_way_stop_scenario_geometry():
    planner, states, graph, config = _planner(GIVE_WAY_STOP)
    roles = dict(graph.get_roles(0))
    assert roles == {1: VesselRole.GIVE_WAY}                # target_2 alongside: no encounter
    assert states["target_2"].east == pytest.approx(27.0)   # 90 m at ship scale


def test_search_stops_to_give_way_when_starboard_is_blocked():
    planner, states, graph, _ = _planner(GIVE_WAY_STOP)
    best = planner.plan(states, graph, t=0.0, responding_probs={"target_1": 0.99,
                                                                "target_2": 0.99}).best
    assert best.actions[0].name == "stop"


def test_give_way_stop_search_clears_both_vessels():
    (traj, *_), log, _, vcfgs, _ = _simulate(GIVE_WAY_STOP, "search_dp")
    ego = traj["ego"]
    for other in ("target_1", "target_2"):
        d = min(math.hypot(p.north - q.north, p.east - q.east) for p, q in zip(ego, traj[other]))
        assert d >= vcfgs["ego"]["d_pass"], other
    assert min(s.speed for s in ego) < 0.1                  # the hull really stopped
    assert log["ego"].plans[0].result.best.actions[0].name == "stop"


# ---------------------------------------------------------------------------
# Risk floor: unlikely hypotheses stay out of the worst case (Rule 17)
# ---------------------------------------------------------------------------

CROSSING = REPO / "scenarios/simulation/milliampere/crossing.yaml"


def test_stand_on_holds_while_give_way_vessel_is_trusted():
    planner, states, graph, _ = _planner(CROSSING)
    assert dict(graph.get_roles(0)) == {1: VesselRole.STAND_ON}
    best = planner.plan(states, graph, t=0.0, responding_probs={"target_1": 0.99}).best
    assert best.actions[0].is_nominal


def test_stand_on_acts_once_give_way_vessel_is_doubted():
    planner, states, graph, _ = _planner(CROSSING)
    best = planner.plan(states, graph, t=0.0, responding_probs={"target_1": 0.5}).best
    assert not best.actions[0].is_nominal


def test_risk_floor_excludes_unlikely_hypothesis_from_worst_case():
    planner, states, graph, _ = _planner(CROSSING)
    nominal = next(a for a in DEFAULT_ACTIONS if a.is_nominal)
    hold = {p.actions: p for p in planner.plan(
        states, graph, t=0.0, responding_probs={"target_1": 0.99}).ranked}[(nominal, nominal)]
    # safety = 0.99 s_trusted + 0.01 s_distrusted; with the floor the worst
    # case is s_trusted alone.
    s_trusted = hold.terms["risk"]
    s_distrusted = (hold.terms["safety"] - 0.99 * s_trusted) / 0.01
    assert s_distrusted > 100 * max(s_trusted, 1e-3)      # holding is a collision if distrusted

    planner.cfg.weights.risk_floor = 0.0                  # old behaviour: max over both
    planner._active = None
    hold_all = {p.actions: p for p in planner.plan(
        states, graph, t=0.0, responding_probs={"target_1": 0.99}).ranked}[(nominal, nominal)]
    assert hold_all.terms["risk"] == pytest.approx(s_distrusted, rel=1e-6)


def test_stopping_never_turns_the_ship_round():
    # Regression: the hull overshoots a planned stop and DP backs it up. That
    # sternway used to be reported as the ship pointing astern, the next plan
    # started from that heading, and DP turned the hull through 180 deg.
    from src.motion_models import adapters

    headings = []
    orig = adapters.MilliAmpereAdapter.step

    def spy(self, action, dt):
        st = orig(self, action, dt)
        headings.append(self._s.xx[2])
        return st

    adapters.MilliAmpereAdapter.step = spy
    try:
        (traj, *_), log, *_ = _simulate(GIVE_WAY_STOP, "search_dp")
    finally:
        adapters.MilliAmpereAdapter.step = orig
    assert any(r.result.best.actions[0].name == "stop" for r in log["ego"].plans)
    assert max(abs(h) for h in headings) < math.radians(45)
    assert all(abs(s.psi) < math.radians(45) for s in traj["ego"])
