"""
Equivalence tests for the scalar (float-tuple) VO solver in
colregs_planner/policies/vo_analytic.py against the frozen numpy implementation in
tests/reference_vo_solver.py.

The two differ only in floating-point evaluation order (numpy's
(N, 2) @ (2,) goes through BLAS, which may use fused multiply-add), so
results are compared with a tight tolerance and the case (1/2/3) must match.

achievable_ratio = 1 is excluded from the strict check: there the feasible
set is a single heading, so rounding alone can decide between case 1 and 2
(see VOPolicyAnalytic's achievable_ratio docstring).  It gets a separate,
looser check that such flips stay rare.

Run with pytest, or directly:  python -m tests.test_vo_solver
"""

from __future__ import annotations

import math
import random

import numpy as np

from colregs_planner.policies import vo_analytic as new
from tests import reference_vo_solver as ref

TOL = 1e-9


def _unit(rng):
    a = rng.uniform(-math.pi, math.pi)
    return np.array([math.cos(a), math.sin(a)])


def _random_problem(rng, ratios=(0.9, 0.95, 0.99)):
    """Random v_pref / obstacles / budget, biased towards conflicts."""
    speed = rng.uniform(0.5, 8.0)
    psi   = rng.uniform(-math.pi, math.pi)
    u     = np.array([math.cos(psi), math.sin(psi)])
    los   = psi + rng.gauss(0.0, 0.1)
    v_pref = speed * np.array([math.cos(los), math.sin(los)])
    obstacles = []
    for _ in range(rng.randint(1, 4)):
        axis  = _unit(rng)
        dist  = rng.uniform(20.0, 3000.0)
        apex  = rng.uniform(0.0, 9.0) * _unit(rng)
        D     = rng.uniform(10.0, 400.0)
        sb    = (axis, apex + rng.gauss(0.0, 0.5) * _unit(rng)) if rng.random() < 0.5 else None
        obstacles.append((apex, axis, dist, D, sb))
    v_prev = (None if rng.random() < 0.2
              else speed * np.array([math.cos(psi + rng.gauss(0, 0.5)), math.sin(psi + rng.gauss(0, 0.5))]))
    return dict(
        v_pref            = v_pref,
        obstacles         = obstacles,
        nominal_unit      = u,
        course_budget     = math.radians(rng.choice([30.0, 60.0, 90.0, 180.0])),
        achievable_ratio  = rng.choice(ratios),
        v_prev            = v_prev,
        continuity_weight = rng.choice([0.0, 0.5, 1.0, 2.0]),
    )


def _close(a, b):
    if isinstance(a, float) and math.isnan(a):
        return math.isnan(b)
    return abs(a - b) <= TOL * max(1.0, abs(a))


def test_solver_matches_reference():
    cases = {1: 0, 2: 0, 3: 0}
    for seed in range(20_000):
        p = _random_problem(random.Random(seed))
        v1, r1, d1, c1 = ref.solve_vo_achievable(**p)
        v2, r2, d2, c2 = new.solve_vo_achievable(**p)
        assert c1 == c2, (seed, c1, c2)
        assert np.allclose(v1, v2, rtol=0, atol=TOL * max(1.0, float(np.abs(v1).max()))), (seed, v1, v2)
        assert len(r1) == len(r2) and all(_close(a, b) for a, b in zip(r1, r2)), (seed, r1, r2)
        assert _close(d1, d2), (seed, d1, d2)
        cases[c1] += 1
    # Guard against a vacuous pass: every branch must be exercised.
    assert all(n > 50 for n in cases.values()), cases


def test_solver_alpha_one_rarely_differs():
    n, n_diff = 5_000, 0
    for seed in range(n):
        p = _random_problem(random.Random(seed), ratios=(1.0,))
        v1, _, _, c1 = ref.solve_vo_achievable(**p)
        v2, _, _, c2 = new.solve_vo_achievable(**p)
        if c1 != c2 or not np.allclose(v1, v2, rtol=0, atol=1e-6):
            n_diff += 1
    assert n_diff <= n // 500, n_diff     # observed: ~0.03%


def test_achievable_pass_distance_matches_reference():
    for seed in range(20_000):
        rng  = random.Random(seed)
        axis = _unit(rng)
        apex = rng.uniform(0.0, 9.0) * _unit(rng)
        sb   = (axis, apex + rng.gauss(0.0, 0.5) * _unit(rng)) if rng.random() < 0.5 else None
        args = (apex, axis, rng.uniform(20.0, 3000.0), rng.uniform(0.5, 8.0), _unit(rng),
                math.radians(rng.choice([30.0, 60.0, 180.0])), sb)
        d1 = ref.achievable_pass_distance(*args)
        d2 = new.achievable_pass_distance(*args)
        assert (d1 is None) == (d2 is None), seed
        if d1 is not None:
            assert _close(d1, d2), (seed, d1, d2)


if __name__ == "__main__":
    test_achievable_pass_distance_matches_reference()
    test_solver_matches_reference()
    test_solver_alpha_one_rarely_differs()
    print("ok")
