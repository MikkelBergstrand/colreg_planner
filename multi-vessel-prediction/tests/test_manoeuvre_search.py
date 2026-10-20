"""
Tests for the manoeuvre-search planner (colregs_planner/planner_api/manoeuvre_search.py),
the search_dp policy, and their wiring into the simulation and the plot.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from colregs_planner.colregs_graph.colregs_classifier import ColregsClassifier
from colregs_planner.colregs_graph.colregs_graph import VesselRole
from colregs_planner.planner_api.manoeuvre_search import (
    COST_CATEGORIES, DEFAULT_ACTIONS, HullModel, ManoeuvreSearchPlanner, SearchAction, SearchConfig,
    SearchWeights, cost_by_category, interpolate_track,
)
from colregs_planner.policies.search_dp import SearchDPPolicy
from colregs_planner.prediction.vo_rollout import rank_hypotheses, vo_rollout
from colregs_planner.scenario_config import build_vo_agent_config, load_config
from colregs_planner.seachart.hazard import HazardMap
from colregs_planner.vessel.state import Intent, VesselState

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


def test_cost_categories_cover_every_term_once():
    planner, states, graph, _ = _planner(horizon_s=40.0)
    result = planner.plan(states, graph, t=0.0)
    grouped = [t for terms, _ in COST_CATEGORIES.values() for t in terms]
    assert sorted(grouped) == sorted(result.best.terms)
    for plan in result.ranked:
        assert sum(cost_by_category(plan.terms).values()) == pytest.approx(plan.cost)
    weight_names = {f for _, names in COST_CATEGORIES.values() for f in names}
    assert weight_names <= set(vars(SearchWeights()))


@pytest.mark.parametrize("probs", [None, {"target_1": 0.99}, {"target_1": 0.0}])
def test_head_on_alters_to_starboard_now(probs):
    # Rule 14 (both alter to starboard) and Rule 16 (early action): even when
    # the target is sure to give way, holding course on the first edge costs.
    planner, states, graph, _ = _planner()
    best = planner.plan(states, graph, t=0.0, responding_probs=probs).best
    assert best.actions[0].course_offset > 0
    assert not any(a.course_offset < 0 for a in best.actions)


def test_own_ship_is_always_responsive():
    """The own ship is under our control: told it is non-responsive, it still
    gets the full search, and every hypothesis has it responsive."""
    planner, states, graph, _ = _planner()
    result = planner.plan(states, graph, t=0.0, intents={"ego": Intent.NON_RESPONSIVE})
    assert len(result.ranked) == len(DEFAULT_ACTIONS) ** planner.cfg.depth
    assert all(a["ego"] == Intent.RESPONSIVE for a, _ in result.hypotheses)


def test_search_dp_rejects_a_non_responsive_own_ship():
    with pytest.raises(ValueError, match="non_responsive"):
        _simulate(REPO / "scenarios/simulation/milliampere/multi_crossing.yaml", "search_dp")


def test_hypotheses_pin_own_intent_and_renormalise():
    planner, states, graph, _ = _planner(horizon_s=20.0)
    result = planner.plan(states, graph, t=0.0,
                          responding_probs={"ego": 0.2, "target_1": 0.7})
    assert len(result.hypotheses) == 2
    assert all(a["ego"] == Intent.RESPONSIVE for a, _ in result.hypotheses)
    assert sum(p for _, p in result.hypotheses) == pytest.approx(1.0)
    assert [p for _, p in result.hypotheses] == pytest.approx([0.7, 0.3])


def test_plan_in_force_continues_edge_by_edge():
    planner, *_ = _planner(edge_s=15.0)
    nominal, a, b = DEFAULT_ACTIONS[0], DEFAULT_ACTIONS[1], DEFAULT_ACTIONS[2]
    assert planner._continuation(0.0) is None
    planner._kept = ((a, b), 10.0)
    assert planner._continuation(20.0) == (a, b)               # first edge under way
    assert planner._continuation(25.0) == (b, nominal)
    assert planner._continuation(60.0) == (nominal, nominal)


def _scored(cost, cpa=0.0):
    from colregs_planner.planner_api.manoeuvre_search import ScoredPlan
    return ScoredPlan(actions=(), cost=cost, min_dist={}, terms={"safety": cpa, "risk": 0.0, "ground": 0.0})


def test_keep_unless_clearly_cheaper_or_less_safe():
    planner, *_ = _planner()
    w = planner.cfg.weights
    kept = _scored(10.0)
    margin = max(w.hold_abs_margin, w.hold_rel_margin * kept.cost)
    assert planner._keep(kept, _scored(10.0 - 0.9 * margin))
    assert not planner._keep(kept, _scored(10.0 - 1.1 * margin))
    # Safety is the limit: a less safe kept plan goes, whatever the margin.
    assert not planner._keep(_scored(10.0, cpa=1.0), _scored(9.99, cpa=0.5))


def test_replan_continues_the_plan_in_force():
    planner, states, graph, _ = _planner(horizon_s=40.0)
    first = planner.plan(states, graph, t=0.0)
    assert first.kept is None
    second = planner.plan(states, graph, t=5.0)          # still on the first edge
    assert second.kept is not None and second.kept.actions == first.best.actions
    assert second.kept in second.ranked
    assert second.best is second.kept or planner._keep(second.kept, second.ranked[0]) is False


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


def test_search_weights_from_yaml():
    sys.path.insert(0, str(REPO / "scripts"))
    import run_simulation as rs
    cfg = rs._search_config({"search": {"edge_s": 10.0, "weights": {"hold_rel_margin": 0.3}}})
    assert cfg.edge_s == 10.0 and cfg.weights.hold_rel_margin == 0.3
    assert cfg.weights.w_dev == SearchWeights().w_dev
    with pytest.raises(ValueError, match="hold_margin"):
        rs._search_config({"search": {"weights": {"hold_margin": 0.3}}})

    # search_dp defaults: the milliAmpere hull model and a passing margin.
    assert cfg.hull == HullModel() and cfg.weights.pass_margin == rs._SEARCH_DP_PASS_MARGIN
    off = rs._search_config({"search": {"hull": False, "weights": {"pass_margin": 0.0}}})
    assert off.hull is None and off.weights.pass_margin == 0.0
    assert rs._search_config({"search": {"hull": {"a_max": 0.1}}}).hull.a_max == 0.1


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
    from colregs_planner.prediction.vo_rollout import VOAgentConfig, vo_rollout

    seq = np.vstack([np.zeros((10, 2)), np.tile([1.5, 0.0], (10, 1))])
    cfg = replace(VOAgentConfig(nominal_psi=0.0, ref_speed=1.5, origin_north=0.0,
                                origin_east=0.0), pref_vel_seq=seq)
    tr = vo_rollout({"ego": VesselState(0.0, 0.0, 0.0, 1.5)}, 1.0, 20, {"ego": cfg},
                    fixed_plan_ids={"ego"})["ego"]
    assert [round(float(tr[k].north), 6) for k in (0, 5, 10)] == [0.0, 0.0, 0.0]
    assert tr[15].north == pytest.approx(7.5)
    assert all(s.psi == 0.0 for s in tr)                  # heading kept while stopped


EAST_BANK = [[-100.0, 14.0], [300.0, 14.0], [300.0, 60.0], [-100.0, 60.0]]


def test_hazard_map_signed_distance():
    hz = HazardMap.from_polygons([EAST_BANK], resolution=1.0)
    d = hz.distance(np.array([[0.0, 0.0], [0.0, 10.0], [50.0, 14.0], [50.0, 20.0], [0.5, 9.5]]))
    assert d == pytest.approx([14.0, 4.0, 0.0, -6.0, 4.5])
    edge = hz.origin[0] + hz.resolution * (hz.sdf.shape[0] - 1)
    assert hz.distance(np.array([[5000.0, 0.0]])) == pytest.approx(hz.distance(np.array([[edge, 0.0]])))


def test_terrain_cost_keeps_plans_off_the_bank():
    plain, states, graph, _ = _planner(hull=HullModel())
    canal, *_ = _planner(hull=HullModel(), hazards=HazardMap.from_polygons([EAST_BANK]))
    a = plain.plan(states, graph, t=0.0)
    b = canal.plan(states, graph, t=0.0)
    assert all(p.terms["ground"] == 0.0 and p.min_shore == math.inf for p in a.ranked)
    d_shore = canal.cfg.weights.d_shore
    # Without the bank the planner turns hard to starboard, which runs aground.
    assert min(p.min_shore for p in b.ranked if p.actions == a.best.actions) < 0.0
    # With it, the best plan keeps (about) d_shore: the cost is soft.
    assert b.best.min_shore > 0.75 * d_shore and b.best.terms["ground"] < 1.0
    assert all(p.terms["ground"] > 0.0 for p in b.ranked if p.min_shore < d_shore)


def test_canal_head_on_stays_off_the_bank():
    (traj, *_), log, config, *_ = _simulate(REPO / "scenarios/simulation/milliampere/canal_head_on.yaml",
                                            "search_dp")
    hz = HazardMap.from_polygons(config.hazards)
    ego = np.array([[s.north, s.east] for s in traj["ego"]])
    assert hz.distance(ego).min() > 0.5 * log["ego"].planner.cfg.weights.d_shore


def test_hull_model_overshoots_a_stop():
    # Reference at 1.5 m/s that stops dead at t = 10 s: the hull runs on.
    t = np.arange(41.0)
    ref = np.stack([1.5 * np.minimum(t, 10.0), np.zeros_like(t)], axis=1)[None]
    pos, vel = HullModel().track(ref, [1.5, 0.0], 1.0)
    overshoot = pos[0, :, 0] - 15.0
    assert 4.0 < overshoot[-1] < 12.0                     # DP stops metres past the reference
    assert np.all(np.diff(pos[0, :, 0]) >= -1e-9)          # without backing up onto it
    assert np.linalg.norm(vel[0, -1]) < 0.05


def test_hull_model_scores_the_hull_track_and_keeps_the_reference():
    planner, states, graph, _ = _planner(edge_s=15.0, plan_dt=1.0, horizon_s=60.0, hull=HullModel())
    result = planner.plan(states, graph, t=0.0)
    hull = result.hull_traj
    assert hull is not None and len(hull) == len(result.own_traj)
    # Every rollout replays the hull track for the own ship.
    for ro in result.rollouts:
        assert [(s.north, s.east) for s in ro["ego"]] == [(s.north, s.east) for s in hull]
    # The reference (what DP follows) is the plan without the hull's lag.
    seq = replace(planner.configs["ego"], pref_vel_seq=planner.velocity_sequence(result.best.actions))
    ref = vo_rollout({"ego": states["ego"]}, 1.0, 60, {"ego": seq}, fixed_plan_ids={"ego"})["ego"]
    assert [(s.north, s.east) for s in result.own_traj] == [(s.north, s.east) for s in ref]


def test_pass_margin_charges_plans_that_only_just_clear_d_pass():
    plain, states, graph, _ = _planner(edge_s=15.0, plan_dt=1.0, horizon_s=60.0)
    margin, *_ = _planner(edge_s=15.0, plan_dt=1.0, horizon_s=60.0,
                          weights=SearchWeights(pass_margin=10.0))
    a = {p.actions: p for p in plain.plan(states, graph, t=0.0).ranked}
    b = {p.actions: p for p in margin.plan(states, graph, t=0.0).ranked}
    d_pass = plain.configs["ego"].d_pass
    assert all(b[k].terms["safety"] >= a[k].terms["safety"] - 1e-9 for k in a)
    clear = [k for k in a if d_pass <= min(a[k].min_dist.values()) < d_pass + 10.0]
    assert clear and all(a[k].terms["safety"] == 0.0 < b[k].terms["safety"] for k in clear)


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


def test_time_cost_does_not_depend_on_edge_length():
    """Time lost is charged per time_ref_s (15 s), not per edge: stopping for
    one 7.5 s edge costs half of stopping for one 15 s edge."""
    planner, states, graph, _ = _planner(edge_s=7.5, plan_dt=1.0, horizon_s=60.0)
    stop = next(a for a in DEFAULT_ACTIONS if a.name == "stop")
    nominal = next(a for a in DEFAULT_ACTIONS if a.is_nominal)
    by_actions = {p.actions: p for p in planner.plan(states, graph, t=0.0).ranked}
    # About 1 s more is lost getting back up to speed.
    assert by_actions[(stop, nominal)].terms["time"] == pytest.approx(0.5, abs=0.1)
    assert by_actions[(stop, stop)].terms["time"] == pytest.approx(1.0, abs=0.1)


def test_effort_scales_with_how_long_an_alteration_is_held():
    """Effort counts per effort_ref_s (15 s): two 60-degree edges cost 2 at
    15 s steps and 1 at 7.5 s steps."""
    stbd_60 = next(a for a in DEFAULT_ACTIONS if a.name == "stbd_60")
    for edge_s, expected in ((15.0, 2.0), (7.5, 1.0)):
        planner, states, graph, _ = _planner(edge_s=edge_s)
        by_actions = {p.actions: p for p in planner.plan(states, graph, t=0.0).ranked}
        assert by_actions[(stbd_60, stbd_60)].terms["effort"] == pytest.approx(expected)


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
    planner.reset()
    hold_all = {p.actions: p for p in planner.plan(
        states, graph, t=0.0, responding_probs={"target_1": 0.99}).ranked}[(nominal, nominal)]
    assert hold_all.terms["risk"] == pytest.approx(s_distrusted, rel=1e-6)


def test_stopping_never_turns_the_ship_round():
    # Regression: the hull overshoots a planned stop and DP backs it up. That
    # sternway used to be reported as the ship pointing astern, the next plan
    # started from that heading, and DP turned the hull through 180 deg.
    from colregs_planner.motion_models import adapters

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


# ---------------------------------------------------------------------------
# Overtaking (Rules 13, 16)
# ---------------------------------------------------------------------------

OVERTAKING_AHEAD = REPO / "scenarios/simulation/milliampere/overtaking_ahead.yaml"


def _closer(states, planner, tcpa_s):
    """Move ego up behind target_1 so the route-course TCPA is tcpa_s."""
    closing = planner.configs["ego"].ref_speed - states["target_1"].speed
    gap = states["target_1"].north - closing * tcpa_s
    return {**states, "ego": replace(states["ego"], north=gap)}


def test_overtaking_holds_course_beyond_the_tcpa_threshold():
    """Out of the predictor's TCPA threshold, the vessel being overtaken
    neither triggers the COLREGs term nor costs passing distance (only a
    collision would): no manoeuvre yet."""
    planner, states, graph, _ = _planner(OVERTAKING_AHEAD)
    assert dict(graph.get_roles(0)) == {1: VesselRole.OVERTAKING}
    assert planner.configs["ego"].tcpa_threshold < 150.0    # starts at TCPA 200 s
    result = planner.plan(states, graph, t=0.0, responding_probs={"target_1": 0.99})
    assert result.best.actions[0].is_nominal
    assert all(p.terms["colregs"] == 0.0 and p.terms["safety"] == 0.0 for p in result.ranked)


def test_overtaking_alters_course_inside_the_tcpa_threshold():
    """Inside the threshold the overtaking own ship alters course, rather than
    holding on or slowing behind the target, although the pass is beyond the
    90 s horizon."""
    planner, states, graph, _ = _planner(OVERTAKING_AHEAD)
    states = _closer(states, planner, 0.9 * planner.configs["ego"].tcpa_threshold)
    result = planner.plan(states, graph, t=0.0, responding_probs={"target_1": 0.99})
    assert result.best.actions[0].course_offset != 0.0
    # Holding course: the CPA after the horizon counts, so it is not "safe".
    nominal = next(a for a in DEFAULT_ACTIONS if a.is_nominal)
    hold = next(p for p in result.ranked if p.actions == (nominal, nominal))
    assert hold.min_dist["target_1"] < 5.0
    assert hold.terms["colregs"] > 0.0


def test_no_overtaking_cost_once_the_offset_track_passes_clear():
    planner, states, graph, _ = _planner(OVERTAKING_AHEAD)
    d_pass = planner.configs["ego"].d_pass
    states = _closer(states, planner, 0.9 * planner.configs["ego"].tcpa_threshold)
    states["ego"] = replace(states["ego"], east=1.2 * d_pass)    # already on an offset track
    result = planner.plan(states, graph, t=0.0, responding_probs={"target_1": 0.99})
    # No conflict, so only turning back toward the target costs anything.
    for p in result.ranked:
        toward = any(a.course_offset < 0 for a in p.actions)
        assert (p.terms["colregs"] > 0.0) == toward
    # Holds its side until past the target.
    assert result.best.actions[0].course_offset >= 0.0
    assert result.best.min_dist["target_1"] >= d_pass


def test_search_dp_overtakes_inside_the_tcpa_threshold_and_passes_clear():
    from colregs_planner.colregs_graph.colregs_classifier import _compute_dcpa_tcpa
    (traj, *_), log, config, vcfgs, _ = _simulate(OVERTAKING_AHEAD, "search_dp")
    ego, tgt = traj["ego"], traj["target_1"]
    d_pass, tcpa_max = vcfgs["ego"]["d_pass"], vcfgs["ego"]["tcpa_threshold"]
    plans = log["ego"].plans
    first = next(r for r in plans if not r.result.best.actions[0].is_nominal)
    # Not before the TCPA threshold (route course, as the planner sees it) ...
    k = round(first.t / config.dt)
    on_route = replace(ego[k], psi=0.0, speed=vcfgs["ego"]["speed"])
    assert _compute_dcpa_tcpa(on_route, tgt[k])[1] < tcpa_max
    # ... and then by altering course, never slowing behind the target.
    assert first.result.best.actions[0].course_offset != 0.0
    assert not any(r.result.best.actions[0].speed_factor < 1.0 for r in plans)
    d = [math.hypot(p.north - q.north, p.east - q.east) for p, q in zip(ego, tgt)]
    assert min(d) >= d_pass
    # Keeps to the starboard side it chose: no plan turns back toward the
    # target before drawing level with it.
    level = next(k for k, (p, q) in enumerate(zip(ego, tgt)) if p.north >= q.north) * config.dt
    assert not any(a.course_offset < 0 for r in plans if r.t < level for a in r.result.best.actions)


def test_overtaken_vessel_ahead_is_not_past_while_turning_away():
    """Mid-alteration, heading well to starboard, the range to the vessel
    being overtaken opens on the actual velocity (TCPA < 0), but it is still
    ahead: turning back toward it keeps costing, and holding on the offset
    track counts its CPA beyond the horizon."""
    from colregs_planner.colregs_graph.colregs_classifier import _compute_dcpa_tcpa
    planner, states, graph, _ = _planner(OVERTAKING_AHEAD)
    d_pass = planner.configs["ego"].d_pass
    states = _closer(states, planner, 0.7 * planner.configs["ego"].tcpa_threshold)
    states["ego"] = replace(states["ego"], east=0.5 * d_pass, psi=math.radians(60.0))
    assert _compute_dcpa_tcpa(states["ego"], states["target_1"])[1] < 0.0
    result = planner.plan(states, graph, t=0.0, responding_probs={"target_1": 0.99})
    toward = [p for p in result.ranked if any(a.course_offset < 0 for a in p.actions)]
    assert all(p.terms["colregs"] >= planner.cfg.weights.w_port for p in toward)
    assert all(a.course_offset >= 0.0 for a in result.best.actions)


def test_overtaking_does_not_turn_back_toward_the_target_near_d_pass():
    """Just inside d_pass on the starboard side, a port turn would also count
    as "altering course" and cost less deviation. It must not be chosen:
    only a turn away from the target satisfies the overtaking rule."""
    planner, states, graph, _ = _planner(OVERTAKING_AHEAD)
    d_pass = planner.configs["ego"].d_pass
    states = _closer(states, planner, 0.7 * planner.configs["ego"].tcpa_threshold)
    states["ego"] = replace(states["ego"], east=0.97 * d_pass)
    result = planner.plan(states, graph, t=0.0, responding_probs={"target_1": 0.99})
    assert result.best.actions[0].course_offset > 0.0
    assert all(a.course_offset >= 0.0 for a in result.best.actions)
    toward = [p for p in result.ranked if any(a.course_offset < 0 for a in p.actions)]
    assert toward and all(p.terms["colregs"] >= planner.cfg.weights.w_port for p in toward)


# ---------------------------------------------------------------------------
# Stand-on toward one vessel, give-way toward others
# ---------------------------------------------------------------------------

MULTI_CROSSING_GIVE_WAY = REPO / "scenarios/simulation/milliampere/multi_crossing_give_way.yaml"


def test_give_way_duty_overrides_stand_on_toward_another_vessel():
    """Give-way to two starboard crossers and stand-on to a port crosser: the
    give-way manoeuvre starts now, not on "the next edge" at every replan."""
    planner, states, graph, _ = _planner(MULTI_CROSSING_GIVE_WAY)
    roles = dict(graph.get_roles(0))
    assert sorted(r.name for r in roles.values()) == ["GIVE_WAY", "GIVE_WAY", "STAND_ON"]
    result = planner.plan(states, graph, t=0.0)
    assert result.best.actions[0].course_offset > 0.0
    starboard_now = [p for p in result.ranked if p.actions[0].course_offset > 0
                     and all(a.course_offset >= 0 for a in p.actions)]
    assert all(p.terms["colregs"] == 0.0 for p in starboard_now)


def test_search_dp_multi_crossing_give_way_passes_clear():
    (traj, *_), log, config, vcfgs, _ = _simulate(MULTI_CROSSING_GIVE_WAY, "search_dp")
    d_pass = vcfgs["ego"]["d_pass"]
    assert log["ego"].plans[0].result.best.actions[0].course_offset > 0.0
    ego = traj["ego"]
    for vid in ("target_1", "target_2", "target_3"):
        d = min(math.hypot(p.north - q.north, p.east - q.east) for p, q in zip(ego, traj[vid]))
        assert d >= d_pass, vid


def test_give_way_cost_lapses_once_the_route_course_passes_clear():
    """Holding course while give-way is only penalised while resuming the
    route course would pass inside d_pass of a give-way vessel."""
    from colregs_planner.colregs_graph.colregs_classifier import _compute_dcpa_tcpa
    planner, states, graph, _ = _planner(MULTI_CROSSING_GIVE_WAY)
    d_pass = planner.configs["ego"].d_pass
    nominal = next(a for a in DEFAULT_ACTIONS if a.is_nominal)

    def hold_colregs(st):
        planner.reset()
        r = planner.plan(st, graph, t=0.0)
        return next(p for p in r.ranked if p.actions == (nominal, nominal)).terms["colregs"]

    assert hold_colregs(states) == planner.cfg.weights.w_give_way
    # Same roles (graph unchanged), but both starboard crossers now pass well ahead.
    clear = {**states, **{vid: replace(states[vid], north=states[vid].north + 4 * d_pass)
                          for vid in ("target_1", "target_2")}}
    for vid in ("target_1", "target_2"):
        dcpa, tcpa = _compute_dcpa_tcpa(clear["ego"], clear[vid])
        assert dcpa >= d_pass and tcpa > 0
    assert hold_colregs(clear) == 0.0


def test_head_on_cost_lapses_once_the_route_course_passes_clear():
    """Not turning to starboard while head-on is only penalised while
    resuming the route course would pass inside d_pass of the vessel."""
    from colregs_planner.colregs_graph.colregs_classifier import _compute_dcpa_tcpa
    planner, states, graph, _ = _planner()
    assert dict(graph.get_roles(0)) == {1: VesselRole.HEAD_ON}
    d_pass = planner.configs["ego"].d_pass
    nominal = next(a for a in DEFAULT_ACTIONS if a.is_nominal)

    def hold_colregs(st):
        planner.reset()
        r = planner.plan(st, graph, t=0.0)
        return next(p for p in r.ranked if p.actions == (nominal, nominal)).terms["colregs"]

    assert hold_colregs(states) == planner.cfg.weights.w_give_way
    # Same role (graph unchanged), but the oncoming vessel now passes well clear.
    clear = {**states, "target_1": replace(states["target_1"], east=states["target_1"].east + 2 * d_pass)}
    dcpa, tcpa = _compute_dcpa_tcpa(clear["ego"], clear["target_1"])
    assert dcpa >= d_pass and tcpa > 0
    assert hold_colregs(clear) == 0.0


# ---------------------------------------------------------------------------
# Crossing ahead of a vessel (Rule 15)
# ---------------------------------------------------------------------------

def _crossing_tracks(along_at_crossing, n=21):
    """Other vessel at the origin heading north (holding position, under way);
    the own ship crosses its heading line west to east, along_at_crossing
    metres ahead of it (negative: astern)."""
    other = [VesselState(north=0.0, east=0.0, psi=0.0, speed=1.0) for _ in range(n)]
    own = np.array([[along_at_crossing, -50.0 + 5.0 * k] for k in range(n)])
    return own, other


def test_bow_crossing_shortfall():
    planner, *_ = _planner()
    d_pass = 30.0
    f = planner.cfg.weights.bow_factor
    own, other = _crossing_tracks(20.0)
    assert planner._bow_crossing_shortfall(own, other, d_pass) == pytest.approx(
        (f * d_pass - 20.0) / (f * d_pass))
    own, other = _crossing_tracks(f * d_pass + 5.0)              # far enough ahead
    assert planner._bow_crossing_shortfall(own, other, d_pass) == 0.0
    own, other = _crossing_tracks(-20.0)                         # astern: no requirement
    assert planner._bow_crossing_shortfall(own, other, d_pass) == 0.0
    parallel = np.array([[5.0 * k, 40.0] for k in range(21)])    # never crosses its track
    assert planner._bow_crossing_shortfall(parallel, other, d_pass) == 0.0
    planner.cfg.weights.bow_factor = 1.0                         # never adds to d_pass
    own, other = _crossing_tracks(20.0)
    assert planner._bow_crossing_shortfall(own, other, d_pass) == 0.0
