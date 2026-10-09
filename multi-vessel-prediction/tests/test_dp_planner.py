"""
Tests for ColregsDPPlanner (colregs_planner/planner_api/dp_planner.py): vessel states in,
DP reference out, closed loop with an own ship that follows the reference.
"""

from __future__ import annotations

import math

import pytest

from colregs_planner.planner_api import ColregsDPPlanner, DPReference, Route

D_PASS = 30.0
CONFIG = {"interaction": {"use_responding_filter": True, "top_k_hypotheses": 2,
                          "reclassify_in_rollout": False,
                          "agent": {"d_pass": D_PASS, "tcpa_threshold": 100.0, "los_lookahead": 30.0}}}
CLASSIFIER = {"d_u": 45.0, "dcpa_max": 450.0, "d_a": 3000.0}
DT = 0.5


def _planner(**kw) -> ColregsDPPlanner:
    return ColregsDPPlanner("ego", Route(0.0, 0.0, 0.0, 1.5), DT, config=CONFIG,
                            classifier=CLASSIFIER, **kw)


def _target(n0, e0, course_deg, speed, t):
    c = math.radians(course_deg)
    return {"north": n0 + speed * math.cos(c) * t, "east": e0 + speed * math.sin(c) * t,
            "course_deg": course_deg, "speed": speed}


def _closed_loop(others, duration=90.0):
    """Own ship reaches each next DP reference exactly; others hold course."""
    own = {"north": 0.0, "east": 0.0, "course_deg": 0.0, "speed": 1.5}
    track, refs, dmin = [], [], {vid: math.inf for vid in others}
    with _planner() as planner:
        for k in range(int(duration / DT) + 1):
            t = k * DT
            obs = {"ego": own, **{vid: _target(*spec, t) for vid, spec in others.items()}}
            for vid in others:
                dmin[vid] = min(dmin[vid], math.hypot(obs[vid]["north"] - own["north"],
                                                      obs[vid]["east"] - own["east"]))
            refs.append(planner.update(t, obs))
            nxt = planner.reference(t + DT)
            own = {"north": nxt.north, "east": nxt.east, "course_deg": nxt.heading_deg,
                   "speed": math.hypot(nxt.vn, nxt.ve)}
            track.append(own)
    return track, refs, dmin


def test_head_on_turns_to_starboard_and_passes_clear():
    track, refs, dmin = _closed_loop({"tgt": (120.0, 0.0, 180.0, 1.2)})
    assert isinstance(refs[0], DPReference) and refs[0].replanned
    assert refs[0].plan[0].startswith("stbd")
    assert dmin["tgt"] >= D_PASS
    assert max(p["east"] for p in track) > 10.0                 # altered to starboard
    assert abs(track[-1]["east"]) < max(p["east"] for p in track)  # heading back to the route


def test_vessel_at_anchor_on_the_route_is_avoided():
    _, _, dmin = _closed_loop({"anchor": (60.0, 0.0, 0.0, 0.0)})
    assert dmin["anchor"] >= D_PASS


def test_replans_every_replan_s_and_on_request():
    with _planner(replan_s=5.0) as planner:
        own = {"north": 0.0, "east": 0.0, "course_deg": 0.0, "speed": 1.5}
        flags = [planner.update(k * DT, {"ego": own}).replanned for k in range(21)]
        assert [k * DT for k, f in enumerate(flags) if f] == [0.0, 5.0, 10.0]
        assert planner.update(10.5, {"ego": own}, force_replan=True).replanned
        planner.set_route(Route(0.0, 0.0, 90.0, 1.5))
        ref = planner.update(11.0, {"ego": own})
        assert ref.replanned and ref.plan_t == 11.0


def test_reference_follows_the_committed_plan_between_updates():
    with _planner() as planner:
        own = {"north": 0.0, "east": 0.0, "course_deg": 0.0, "speed": 1.5}
        ref0 = planner.update(0.0, {"ego": own})
        ref = planner.reference(2.0)                             # no traffic: along the route
        assert ref.plan == ref0.plan and ref.plan_t == 0.0
        assert ref.north == pytest.approx(3.0, abs=0.1) and ref.east == pytest.approx(0.0, abs=1e-6)
        assert ref.vn == pytest.approx(1.5, abs=0.05) and ref.heading_deg == pytest.approx(0.0, abs=1e-6)
        far = planner.reference(500.0)                           # past the horizon: extrapolated
        assert far.north > ref.north


def test_input_errors():
    planner = _planner()
    with pytest.raises(RuntimeError):
        planner.reference(0.0)
    with pytest.raises(KeyError, match="ego"):
        planner.update(0.0, {"other": {"north": 0.0, "east": 0.0, "course_deg": 0.0, "speed": 1.0}})
    planner.close()
