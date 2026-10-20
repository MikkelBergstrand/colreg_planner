"""
Tests for the milliAmpere simulation model, its adapter, and the scenario
plumbing added for the scaled milliAmpere scenarios.
"""

from __future__ import annotations

import contextlib
import io
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from colregs_planner.motion_models.adapters import MilliAmpereAdapter, build_adapter
from colregs_planner.motion_models.milliampere_model import MilliAmpereSimulator, _rot, _ssa
from colregs_planner.policies.predictor_dp import PredictorDPPolicy
from colregs_planner.scenario_config import load_config
from colregs_planner.vessel.encounter import EncounterType, _OVERTAKING_LATERAL, _target_init
from colregs_planner.vessel.state import CTAction, CVAction, DPAction, InitialState, VesselState, wrap_angle

REPO = Path(__file__).resolve().parents[1]
MA_SCENARIOS = sorted((REPO / "scenarios/simulation/milliampere").glob("*.yaml"))


def _run(sim: MilliAmpereSimulator, s, psi_cmd: float, speed_cmd: float, seconds: float, dt=0.5):
    states = []
    for _ in range(round(seconds / dt)):
        s = sim.step(s, psi_cmd, speed_cmd, dt)
        states.append(s)
    return s, states


# ---------------------------------------------------------------------------
# MilliAmpereSimulator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("speed", [0.5, 1.0, 1.5])
def test_holds_initial_speed_and_heading(speed):
    sim = MilliAmpereSimulator()
    s = sim.reset(0.0, 0.0, 0.3, speed)
    s, states = _run(sim, s, 0.3, speed, 30.0)

    # The trim-loaded integrator means no initial sag while it winds up.
    assert max(abs(st.xx[3] - speed) for st in states) < 1e-3 * speed
    assert abs(s.xx[2] - 0.3) < 1e-9
    # Straight run along the commanded heading.
    assert math.hypot(s.xx[0], s.xx[1]) == pytest.approx(30.0 * speed, rel=1e-3)
    assert math.atan2(s.xx[1], s.xx[0]) == pytest.approx(0.3, abs=1e-6)


def test_turns_to_commanded_heading():
    sim = MilliAmpereSimulator()
    s = sim.reset(0.0, 0.0, 0.0, 1.5)
    target = math.radians(90)
    s, states = _run(sim, s, target, 1.5, 60.0)

    errs = [abs(wrap_angle(st.xx[2] - target)) for st in states]
    t_within_5deg = next(i for i, e in enumerate(errs) if e < math.radians(5)) * 0.5
    assert 10.0 < t_within_5deg < 30.0, "liteAmpere tuning turns 90 deg in ~19 s"
    assert errs[-1] < math.radians(0.5)
    assert max(abs(st.xx[5]) for st in states) < math.radians(10)   # yaw rate stays sane


def test_turn_across_180_takes_short_way():
    sim = MilliAmpereSimulator()
    s = sim.reset(0.0, 0.0, math.radians(170), 1.0)
    target = math.radians(-170)                      # +20 deg the short way
    s, states = _run(sim, s, target, 1.0, 40.0)

    # Never swings back through north, which a full -340 deg unwind would.
    assert all(abs(wrap_angle(st.xx[2] - math.pi)) < math.radians(35) for st in states)
    assert abs(wrap_angle(s.xx[2] - target)) < math.radians(1)


def test_speed_change_tracks_command():
    # liteAmpere's speed PI (kp=100, ki=10 on 2390 kg) is slow: a 1.0 -> 1.5 m/s
    # step reaches 90 % in ~85 s and 98 % in ~160 s, without overshoot.
    sim = MilliAmpereSimulator()
    s = sim.reset(0.0, 0.0, 0.0, 1.0)
    s, states = _run(sim, s, 0.0, 1.5, 200.0)
    assert s.xx[3] == pytest.approx(1.5, abs=0.01)
    assert max(st.xx[3] for st in states) <= 1.5 + 1e-6


@pytest.mark.parametrize("dt", [0.0, 0.005, 0.255])
def test_rejects_dt_not_multiple_of_inner_dt(dt):
    sim = MilliAmpereSimulator(inner_dt=0.01)
    s = sim.reset(0.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        sim.step(s, 0.0, 1.0, dt)


def test_step_does_not_mutate_input_state():
    sim = MilliAmpereSimulator()
    s0 = sim.reset(1.0, 2.0, 0.5, 1.0)
    xx0, ref0, integ0 = s0.xx.copy(), s0.ref.copy(), s0.speed_integral
    sim.step(s0, 1.0, 1.2, 0.5)
    np.testing.assert_array_equal(s0.xx, xx0)
    np.testing.assert_array_equal(s0.ref, ref0)
    assert s0.speed_integral == integ0


# ---------------------------------------------------------------------------
# Dynamic positioning
# ---------------------------------------------------------------------------

def _litampere_dp_tau(sim, xx, eta_ref):
    """liteAmpere DynamicPositioning.update with Ki = 0 (its default)."""
    e = xx[:3] - eta_ref
    e[2] = _ssa(e[2])
    R = _rot(xx[2])
    return -(R.T @ sim._dp_Kp @ e + R.T @ sim._dp_Kd @ R @ xx[3:6])


def test_dp_with_fixed_reference_is_litampere_law():
    sim = MilliAmpereSimulator()
    s = sim.reset(0.0, 0.0, 0.2, 0.8)
    s.xx[4:6] = [0.1, -0.02]
    eta_ref = np.array([4.0, -2.0, 0.6])

    # One inner step of step_dp against an rk4 step under liteAmpere's tau.
    from colregs_planner.motion_models.milliampere_model import rk4
    tau = _litampere_dp_tau(sim, s.xx.copy(), eta_ref)
    expected = rk4(s.xx.copy(), lambda x: sim.model.derivative(x, tau), sim.inner_dt)
    got = sim.step_dp(s, eta_ref, np.zeros(3), sim.inner_dt)
    np.testing.assert_allclose(got.xx, expected, rtol=0, atol=1e-12)


def test_dp_holds_station():
    # liteAmpere's gains settle slowly (Kd north = 10): ~5 cm off after 120 s,
    # settled to the millimetre by ~6 min.
    sim = MilliAmpereSimulator()
    s = sim.reset(0.0, 0.0, 0.0, 0.0)
    target = np.array([5.0, -3.0, 0.5])
    for _ in range(720):
        s = sim.step_dp(s, target, np.zeros(3), 0.5)
    np.testing.assert_allclose(s.xx[:3], target, atol=0.005)
    assert np.all(np.abs(s.xx[3:]) < 1e-3)


@pytest.mark.parametrize("course_deg", [0.0, 90.0, 225.0])
def test_dp_tracks_moving_reference_without_lag(course_deg):
    # liteAmpere's law as written trails a 1.5 m/s reference by 10-17 m; with
    # velocity-error damping and feedforward the hull stays on it.
    sim = MilliAmpereSimulator()
    chi = math.radians(course_deg)
    rate = np.array([1.5 * math.cos(chi), 1.5 * math.sin(chi), 0.0])
    s = sim.reset(0.0, 0.0, chi, 1.5)
    ref = np.array([0.0, 0.0, chi])
    for _ in range(240):
        s = sim.step_dp(s, ref, rate, 0.5)
        ref = ref + rate * 0.5
    assert math.hypot(*(ref[:2] - s.xx[:2])) < 0.01
    assert math.hypot(s.xx[3], s.xx[4]) == pytest.approx(1.5, abs=1e-3)


def test_dp_tracks_turning_reference():
    sim = MilliAmpereSimulator()
    s = sim.reset(0.0, 0.0, 0.0, 1.5)
    ref, t, lags = np.zeros(3), 0.0, []
    for _ in range(160):
        chi = math.radians(45) * min(max((t - 10) / 10, 0), 1)
        r = math.radians(45) / 10 if 10 <= t < 20 else 0.0
        rate = np.array([1.5 * math.cos(chi), 1.5 * math.sin(chi), r])
        s = sim.step_dp(s, np.array([ref[0], ref[1], chi]), rate, 0.5)
        ref = ref + rate * 0.5
        t += 0.5
        lags.append(math.hypot(*(ref[:2] - s.xx[:2])))
    assert max(lags) < 0.5
    assert abs(_ssa(s.xx[2] - math.radians(45))) < math.radians(1)


# ---------------------------------------------------------------------------
# MilliAmpereAdapter
# ---------------------------------------------------------------------------

def test_build_adapter_milliampere():
    assert isinstance(build_adapter("milliampere", 0.5, k_psi=0.1), MilliAmpereAdapter)


def test_adapter_init_exports_initial_state():
    ad = MilliAmpereAdapter()
    st = ad.init(InitialState(north=10.0, east=-5.0, speed=1.2, heading=math.radians(30)))
    assert (st.north, st.east, st.speed, st.r) == (10.0, -5.0, 1.2, 0.0)
    assert st.psi == pytest.approx(math.radians(30))


def test_adapter_cv_action_steers_to_velocity_heading():
    ad = MilliAmpereAdapter()
    ad.init(InitialState(speed=1.5, heading=0.0))
    hdg = math.radians(45)
    for _ in range(120):
        st = ad.step(CVAction(vn=1.5 * math.cos(hdg), ve=1.5 * math.sin(hdg)), 0.5)
    hull = ad._s.xx[2]
    assert abs(wrap_angle(hull - hdg)) < math.radians(0.5)
    assert st.speed == pytest.approx(1.5, abs=0.05)


def test_adapter_exports_course_over_ground():
    ad = MilliAmpereAdapter()
    ad.init(InitialState(speed=1.5, heading=0.0))
    st = None
    for _ in range(10):                               # mid-turn: large sideslip
        st = ad.step(CVAction(vn=0.0, ve=1.5), 0.5)
    north, east, psi, u, v, _ = ad._s.xx
    vn = u * math.cos(psi) - v * math.sin(psi)
    ve = u * math.sin(psi) + v * math.cos(psi)
    assert abs(math.atan2(v, u)) > math.radians(5), "test needs a turn with sideslip"
    assert st.vn == pytest.approx(vn, abs=1e-9)
    assert st.ve == pytest.approx(ve, abs=1e-9)


def test_adapter_stopped_cv_action_keeps_heading():
    ad = MilliAmpereAdapter()
    ad.init(InitialState(speed=1.0, heading=math.radians(60)))
    ad.step(CVAction(vn=0.0, ve=0.0), 0.5)
    assert ad._psi_cmd == pytest.approx(math.radians(60))


def test_adapter_ct_action_advances_heading_command():
    ad = MilliAmpereAdapter()
    ad.init(InitialState(speed=1.0, heading=0.0))
    for _ in range(10):
        ad.step(CTAction(speed=1.0, turn_rate=0.02), 0.5)
    assert ad._psi_cmd == pytest.approx(10 * 0.02 * 0.5)


def test_adapter_dp_reference_accumulates_steps():
    ad = MilliAmpereAdapter()
    ad.init(InitialState(north=2.0, east=1.0, speed=1.5, heading=0.0))
    np.testing.assert_allclose(ad.dp_reference, [2.0, 1.0, 0.0])
    for _ in range(20):
        st = ad.step(DPAction(d_north=0.75, d_east=0.0, psi_ref=0.0), 0.5)
    np.testing.assert_allclose(ad.dp_reference, [2.0 + 15.0, 1.0, 0.0])
    assert math.hypot(st.north - 17.0, st.east - 1.0) < 0.01


def test_adapter_dp_then_heading_control_continues_smoothly():
    ad = MilliAmpereAdapter()
    ad.init(InitialState(speed=1.5, heading=0.0))
    for _ in range(10):
        ad.step(DPAction(d_north=0.75, d_east=0.0, psi_ref=0.0), 0.5)
    for _ in range(10):
        st = ad.step(CVAction(vn=1.5, ve=0.0), 0.5)
    assert st.speed == pytest.approx(1.5, abs=0.05)
    assert abs(st.psi) < math.radians(1)
    np.testing.assert_allclose(ad.dp_reference, ad._s.xx[:3])


# ---------------------------------------------------------------------------
# PredictorDPPolicy
# ---------------------------------------------------------------------------

def test_predictor_dp_policy_uses_predicted_step_once():
    pol = PredictorDPPolicy(dt=0.5)
    own = VesselState(north=0.0, east=0.0, psi=0.0, speed=1.0)
    pol.set_predicted_step(
        VesselState(north=10.0, east=5.0, psi=0.0, speed=1.0),
        VesselState(north=10.4, east=5.3, psi=0.6, speed=1.0),
    )
    a = pol.select_action(own, [])
    assert (a.d_north, a.d_east, a.psi_ref) == pytest.approx((0.4, 0.3, 0.6))

    # Without a fresh prediction it carries on at the current velocity.
    b = pol.select_action(VesselState(north=0.0, east=0.0, psi=math.pi / 2, speed=2.0), [])
    assert (b.d_north, b.d_east, b.psi_ref) == pytest.approx((0.0, 1.0, math.pi / 2))


# ---------------------------------------------------------------------------
# Scenario plumbing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("enc, sign", [
    (EncounterType.OVERTAKING_AHEAD, 1), (EncounterType.OVERTAKING_ASTERN, -1),
])
def test_overtaking_lateral_offset_default_and_override(enc, sign):
    default = _target_init(enc, 200.0, 5.0, 3.0, 0.0)
    assert (default.north, default.east) == pytest.approx((sign * 200.0, _OVERTAKING_LATERAL))
    custom = _target_init(enc, 200.0, 5.0, 3.0, 0.0, lateral_offset_m=15.0)
    assert (custom.north, custom.east) == pytest.approx((sign * 200.0, 15.0))


def test_head_on_lateral_offset_defaults_to_zero():
    s = _target_init(EncounterType.HEAD_ON, 400.0, 5.0, 4.0, 0.0)
    assert (s.north, s.east) == pytest.approx((400.0, 0.0))


def test_classifier_block_loaded_and_validated(tmp_path):
    base = yaml.safe_load((REPO / "scenarios/simulation/head_on.yaml").read_text())
    assert load_config(REPO / "scenarios/simulation/head_on.yaml")[0].classifier == {}

    base["scenario"]["classifier"] = {"d_u": 45.0, "dcpa_max": 450.0}
    good = tmp_path / "good.yaml"
    good.write_text(yaml.safe_dump(base))
    assert load_config(good)[0].classifier == {"d_u": 45.0, "dcpa_max": 450.0}

    base["scenario"]["classifier"] = {"d_uu": 45.0}
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(base))
    with pytest.raises(ValueError, match="d_uu"):
        load_config(bad)


# ---------------------------------------------------------------------------
# Scaled milliAmpere scenarios: smoke test end to end
# ---------------------------------------------------------------------------

def _simulate(config, vcfgs, pred):
    sys.path.insert(0, str(REPO / "scripts"))
    import run_simulation as rs

    with contextlib.redirect_stdout(io.StringIO()):
        return rs.simulate_scenario(
            config, vcfgs, pred.use_responding_filter,
            pred.filter_rollout_model or pred.rollout_model, pred_cfg=pred,
        )


@pytest.mark.parametrize("path", MA_SCENARIOS, ids=lambda p: p.stem)
def test_milliampere_scenario_runs(path):
    config, vcfgs, _, pred = load_config(path)
    assert vcfgs["ego"]["sim_model"] == "milliampere"
    # A non-responsive ego is a test vessel, not the ship under our control.
    # The terrain and channel scenarios show the search planner (search_dp).
    expected = ("vo_analytic" if vcfgs["ego"].get("intent") == "non_responsive"
                else "search_dp" if config.hazards or config.classifier.get("channels") else "predictor_dp")
    assert vcfgs["ego"]["policy"] == expected
    traj, *_ = _simulate(config, vcfgs, pred)

    ego = traj["ego"]
    assert len(ego) == config.n_steps + 1
    assert all(math.isfinite(s.north) and math.isfinite(s.east) for s in ego)
    assert max(s.speed for s in ego) < 2.0, "ego must stay inside milliAmpere's envelope"
    # No vessel pair collides (hull length is ~5 m).
    ids = list(traj)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            d = min(math.hypot(p.north - q.north, p.east - q.east) for p, q in zip(traj[a], traj[b]))
            assert d > 5.0, f"{a}-{b} closest approach {d:.1f} m"


def test_predictor_dp_requires_milliampere():
    config, vcfgs, _, pred = load_config(REPO / "scenarios/simulation/milliampere/head_on.yaml")
    vcfgs["ego"]["sim_model"] = "ct_decoupled_ar"
    with pytest.raises(ValueError, match="predictor_dp"):
        _simulate(config, vcfgs, pred)


@pytest.mark.parametrize("use_filter", [True, False])
def test_non_responsive_ego_holds_course(use_filter):
    # multi_crossing's ego is the vessel the others must avoid: a plain policy
    # that ignores traffic, on the milliAmpere model.
    path = REPO / "scenarios/simulation/milliampere/multi_crossing.yaml"
    config, vcfgs, _, pred = load_config(path)
    assert vcfgs["ego"]["intent"] == "non_responsive"
    assert vcfgs["ego"]["policy"] == "vo_analytic"
    pred.use_responding_filter = use_filter
    traj, *_ = _simulate(config, vcfgs, pred)
    assert max(abs(s.east) for s in traj["ego"]) < 0.01


def test_predictor_dp_rejects_a_non_responsive_own_ship():
    # A planned vessel is the own ship under our control: always compliant.
    config, vcfgs, _, pred = load_config(REPO / "scenarios/simulation/milliampere/multi_crossing.yaml")
    vcfgs["ego"]["policy"] = "predictor_dp"
    with pytest.raises(ValueError, match="non_responsive"):
        _simulate(config, vcfgs, pred)


def test_predictor_dp_follows_one_step_prediction():
    # Each step's DP reference move equals ego's step in a one-step rollout of
    # the same predictor from that step's states.
    path = REPO / "scenarios/simulation/milliampere/head_on.yaml"
    config, vcfgs, _, pred = load_config(path)

    from colregs_planner.policies import predictor_dp as mod
    seen = []
    orig = mod.PredictorDPPolicy.set_predicted_step

    def spy(self, now, nxt):
        seen.append((now, nxt))
        orig(self, now, nxt)

    mod.PredictorDPPolicy.set_predicted_step = spy
    try:
        traj, *_ = _simulate(config, vcfgs, pred)
    finally:
        mod.PredictorDPPolicy.set_predicted_step = orig

    assert len(seen) == config.n_steps
    for k, (now, _) in enumerate(seen):
        assert (now.north, now.east) == (traj["ego"][k].north, traj["ego"][k].east)
    # The prediction does turn ego away from its straight course at some point.
    assert max(abs(nxt.psi) for _, nxt in seen) > math.radians(5)


def test_planned_vessels_are_pinned_responsive():
    # Every prediction made for a predictor_dp / search_dp vessel (the one it
    # follows and the one drawn) pins it responsive; plain-policy vessels are
    # left to the filter.
    sys.path.insert(0, str(REPO / "scripts"))
    import run_simulation as rs

    for name, expected in (("multi_crossing_give_way", {"ego": 1.0}), ("multi_crossing", {})):
        config, vcfgs, _, _ = load_config(REPO / f"scenarios/simulation/milliampere/{name}.yaml")
        assert rs._dp_own_intent_probs(config, vcfgs) == expected


def test_adapter_reports_hull_heading_when_going_astern():
    # DP backing onto a reference the hull overshot: the course over ground
    # points astern, but the ship has not turned round.
    ad = MilliAmpereAdapter()
    ad.init(InitialState(speed=1.5, heading=0.0))
    astern = 0
    for _ in range(80):
        st = ad.step(DPAction(d_north=0.0, d_east=0.0, psi_ref=0.0), 0.5)
        if ad._s.xx[3] < -0.05:
            astern += 1
            assert st.psi == pytest.approx(0.0, abs=1e-9)
            assert st.speed == 0.0
    assert astern > 5, "test needs sternway"


def test_adapter_dp_ref_start_places_reference():
    ad = MilliAmpereAdapter()
    ad.init(InitialState(speed=1.0, heading=0.0))
    ad.step(DPAction(d_north=0.5, d_east=0.0, psi_ref=0.0, ref_start=(10.0, -2.0, 0.0)), 0.5)
    np.testing.assert_allclose(ad.dp_reference, [10.5, -2.0, 0.0])


def test_rollout_speed_response_is_off_by_default():
    # tau_speed = 0 applies the commanded speed at once; tau_speed > 0 follows
    # a first-order response, v -> v + (v_cmd - v) * (1 - exp(-dt / tau)).
    from colregs_planner.vessel.state import CVAction
    for tau, expected in ((0.0, 0.0), (4.0, 1.5 * math.exp(-1.0 / 4.0))):
        ad = build_adapter("ct_decoupled_ar", 1.0, k_psi=0.1, tau_speed=tau)
        ad.init(InitialState(speed=1.5, heading=0.0))
        st = ad.step(CVAction(vn=0.0, ve=0.0), 1.0)                 # stop command
        assert st.speed == pytest.approx(expected)
