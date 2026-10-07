"""
The parallel rollout path (src/prediction/parallel.py) must give exactly the
same trajectories as the serial one.

Run with pytest, or directly:  python -m tests.test_parallel
(the __main__ guard is required: worker processes import this file).
"""

from __future__ import annotations

import math

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


if __name__ == "__main__":
    test_resolve_workers()
    test_parallel_matches_serial()
    test_rollout_error_propagates_from_worker()
    print("ok")
