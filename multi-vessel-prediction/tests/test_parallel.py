"""
The parallel rollout path (src/prediction/parallel.py) must give exactly the
same trajectories as the serial one.

Run with pytest, or directly:  python -m tests.test_parallel
(the __main__ guard is required: worker processes import this file).
"""

from __future__ import annotations

import math

import pytest

from src.colregs_graph.colregs_classifier import ColregsClassifier
from src.prediction.parallel import RolloutPool, resolve_workers, submit_rollout
from src.prediction.vo_rollout import VOAgentConfig, enumerate_and_rollout
from src.vessel.state import VesselState


def _scene():
    """Head-on pair plus a crossing vessel, all within detection range."""
    states = {
        "ego": VesselState(north=0.0,    east=0.0,    psi=0.0,             speed=5.0),
        "t1":  VesselState(north=1200.0, east=30.0,   psi=math.pi,         speed=4.0),
        "t2":  VesselState(north=500.0,  east=600.0,  psi=-math.pi / 2,    speed=4.0),
    }
    index_map = {"ego": 0, "t1": 1, "t2": 2}
    classifier = ColregsClassifier()
    nom_graph  = classifier.update({index_map[v]: s for v, s in states.items()})
    configs    = {vid: VOAgentConfig() for vid in states}
    return states, index_map, classifier, nom_graph, configs


def _flat(trajs):
    return {vid: [(s.north, s.east, s.psi, s.speed, s.r) for s in ss] for vid, ss in trajs.items()}


def test_parallel_matches_serial():
    states, index_map, classifier, nom_graph, configs = _scene()
    kwargs = dict(
        initial_states   = states,
        responding_probs = {"ego": 0.9, "t1": 0.7, "t2": 0.4},
        nom_graph        = nom_graph,
        dt               = 1.0,
        n_steps          = 120,
        agent_configs    = configs,
        vessel_index_map = index_map,
        top_k            = 8,
        colregs_classifier = classifier,
    )
    serial = enumerate_and_rollout(**kwargs)
    with RolloutPool(n_workers=4) as pool:
        parallel = enumerate_and_rollout(**kwargs, pool=pool)

    assert len(serial) == len(parallel) == 8
    for (t_s, a_s, w_s), (t_p, a_p, w_p) in zip(serial, parallel):
        assert a_s == a_p and w_s == w_p
        assert _flat(t_s) == _flat(t_p)
    # The trajectories must actually differ between hypotheses, or the
    # comparison above proves little.
    assert len({str(_flat(t)) for t, _, _ in serial}) > 1


def test_rollout_error_propagates_from_worker():
    states, index_map, _, _, configs = _scene()
    bad = {vid: VOAgentConfig(ref_speed=0.0) for vid in states}   # vo_rollout raises ValueError
    with RolloutPool(n_workers=1) as pool:
        handle = submit_rollout(pool, initial_states=states, dt=1.0, n_steps=5,
                                agent_configs=bad, vessel_index_map=index_map)
        try:
            handle.result()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError from the worker")


def test_resolve_workers():
    import os
    cores = max(1, (os.cpu_count() or 1) - 1)
    assert resolve_workers(0, 4) == 0
    assert resolve_workers(3, 1) == 3                       # explicit value used as given
    assert resolve_workers("auto", 1) == 0                  # one rollout at a time: no pool
    assert resolve_workers("auto", 4) == (min(4, cores) if min(4, cores) > 1 else 0)
    assert resolve_workers("auto", None) == (cores if cores > 1 else 0)
    try:
        resolve_workers(-1, None)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a negative worker count")


def test_search_planner_pool_matches_serial():
    from tests.test_manoeuvre_search import _planner
    planner, states, graph, config = _planner()
    clf = ColregsClassifier(**config.classifier)
    probs = {"target_1": 0.7}

    def ranked():
        planner._active = None          # no action under way, so no switch cost
        return planner.plan(states, graph, t=0.0, responding_probs=probs, classifier=clf)

    serial = ranked()
    with RolloutPool(n_workers=4) as pool:
        planner.pool = pool
        parallel = ranked()
    # Same trajectories and ranking; costs may differ in the last bit, since
    # pooled rollouts come back as Python floats rather than numpy scalars.
    assert [_flat(t) for t in serial.rollouts] == [_flat(t) for t in parallel.rollouts]
    assert [p.actions for p in serial.ranked] == [p.actions for p in parallel.ranked]
    for s, p in zip(serial.ranked, parallel.ranked):
        assert s.min_dist == p.min_dist
        assert p.cost == pytest.approx(s.cost, rel=1e-12)


def _service(parallel_workers):
    from src.planner_api import PredictionService
    service = PredictionService.from_config(
        {"H": 60, "interaction": {"use_responding_filter": True, "top_k_hypotheses": 4,
                                  "parallel_workers": parallel_workers,
                                  "agent": {"d_pass": 150.0}}},
        dt=2.0, own_id="ego",
    )
    obs = {"ego":    {"north": 0.0,    "east": 0.0,    "course_deg": 0.0,   "speed": 5.0},
           "target": {"north": 1000.0, "east": 1000.0, "course_deg": 270.0, "speed": 5.0}}
    service.step(obs, t=0.0)
    return service


_SPECS = [dict(course_change_deg=c, start_s=d) for c in (0.0, 30.0, -30.0) for d in (0.0, 30.0)]


def test_evaluate_manoeuvres_pool_matches_serial():
    serial = _service(0).evaluate_manoeuvres("ego", _SPECS)
    service = _service(2)
    try:
        assert service.evaluate_manoeuvres("ego", _SPECS) == serial
    finally:
        service.predictor._pool.shutdown()
    single = _service(0).evaluate_manoeuvre("ego", 30.0, start_s=30.0)
    assert single == serial[3]


if __name__ == "__main__":
    test_resolve_workers()
    test_parallel_matches_serial()
    test_rollout_error_propagates_from_worker()
    test_search_planner_pool_matches_serial()
    test_evaluate_manoeuvres_pool_matches_serial()
    print("ok")
