"""
Frozen copy of the numpy-based VO solver from colregs_planner/policies/vo_analytic.py
(before the scalar rewrite), used as the reference in
tests/test_vo_solver.py.  Do not edit.
"""

from __future__ import annotations

import math

import numpy as np

from colregs_planner.policies.vo_common import VOCone


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------
# Boundaries are "primitives" (p, d, ray): the line p + t*d (unit d), or the
# ray t >= 0 when ray is True.  Starboard constraints are (ru, ref): unit
# bearing to the obstacle and the reference velocity of the obstacle.

def _prim_circle(p: np.ndarray, d: np.ndarray, ray: bool, radius: float) -> list[np.ndarray]:
    """Points where line/ray p + t*d (unit d) meets the circle |v| = radius."""
    b    = float(p[0] * d[0] + p[1] * d[1])
    disc = b * b - (float(p[0] * p[0] + p[1] * p[1]) - radius * radius)
    if disc < 0.0:
        return []
    sq = math.sqrt(disc)
    return [p + t * d for t in (-b - sq, -b + sq) if not (ray and t < 0.0)]


def _budget_rays(u: np.ndarray, budget: float) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    """The two course-budget edges as rays from the origin (none if >= 180 deg)."""
    if budget >= math.pi - 1e-9:
        return []
    rays = []
    for sgn in (-1.0, 1.0):
        ca, sa = math.cos(sgn * budget), math.sin(sgn * budget)
        rays.append((np.zeros(2), np.array([ca * u[0] - sa * u[1], sa * u[0] + ca * u[1]]), True))
    return rays


def _boundary_prims(cones, sb_units):
    """Boundary primitives of the cones (both legs) and starboard constraints."""
    prims: list[tuple[np.ndarray, np.ndarray, bool]] = []
    for cone in cones:
        prims.append((cone.apex, cone.left_leg,  True))
        prims.append((cone.apex, cone.right_leg, True))
    for ru, ref in sb_units:
        prims.append((ref, ru, False))                         # side boundary (cross = 0)
        prims.append((ref, np.array([-ru[1], ru[0]]), False))  # converging gate (range rate 0)
    return prims


def _feasible_mask(cands, cones, sb_units, tol, epsilon):
    """True where a candidate velocity is outside every cone and satisfies
    every starboard constraint (not converging on the port side)."""
    ok = np.ones(len(cands), dtype=bool)
    for cone in cones:
        rel    = cands - cone.apex
        dot_ax = rel @ cone.axis
        thr_sq = (cone.cos_half_angle + epsilon) ** 2
        ok    &= ~((dot_ax > 0.0) & (dot_ax * dot_ax > thr_sq * (rel * rel).sum(axis=1)))
    for ru, ref in sb_units:
        rv    = cands - ref
        conv  = rv @ ru > tol
        cross = ru[0] * rv[:, 1] - ru[1] * rv[:, 0]
        ok   &= ~(conv & (cross < -tol))
    return ok


def _ring_pick(v_pref, v_prev, lam, R, u, budget, cones, sb_units, epsilon):
    """Paper Eq. 19 on the speed ring |v| = R within the course budget.

    The minimum of |angle(v, v_pref)| + lam * |angle(v, v_prev)| over the
    feasible arcs lies at v_pref, at v_prev or at an arc end, i.e. where a
    boundary primitive meets the ring -- so only those candidates are
    evaluated.  None if no heading is feasible.
    """
    tol   = 1e-7 * max(1.0, R)
    prims = _boundary_prims(cones, sb_units) + _budget_rays(u, budget)
    ring: list[np.ndarray] = [np.asarray(v_pref, dtype=float)]
    if v_prev is not None:
        ring.append(np.asarray(v_prev, dtype=float) * (R / math.hypot(float(v_prev[0]), float(v_prev[1]))))
    for p, d, ray in prims:
        ring.extend(_prim_circle(p, d, ray, R))
    cands = np.array(ring)
    ok  = _feasible_mask(cands, cones, sb_units, tol, epsilon)
    ok &= (cands @ u) >= math.cos(budget) * R - tol
    if not ok.any():
        return None
    fm  = cands[ok]
    ang = np.arctan2(fm[:, 1], fm[:, 0])
    pa  = math.atan2(float(v_pref[1]), float(v_pref[0]))
    cost = np.abs(np.arctan2(np.sin(ang - pa), np.cos(ang - pa)))
    if v_prev is not None and lam > 0.0:
        qa    = math.atan2(float(v_prev[1]), float(v_prev[0]))
        cost += lam * np.abs(np.arctan2(np.sin(ang - qa), np.cos(ang - qa)))
    return fm[np.argmin(cost)].copy()


# ---------------------------------------------------------------------------
# Cone shift (conflict case)
# ---------------------------------------------------------------------------

def _cone_edge_pieces(cone: VOCone) -> list[tuple[np.ndarray, float]]:
    """Inward unit normals n and offsets c = n . apex of the cone's edges.

    The cone (half-angle <= 90 deg) is the intersection of the half-planes
    n . (v - apex) > 0, so the velocity shift still missing to leave it is
    max(0, min_k (n_k . v - c_k)).  A half-plane (half-angle 90 deg) has a
    single edge.
    """
    pieces: list[tuple[np.ndarray, float]] = []
    for leg in (cone.left_leg, cone.right_leg):
        n = np.array([-leg[1], leg[0]])
        if float(n @ cone.axis) < 0.0:
            n = -n
        if pieces and float(n @ pieces[0][0]) > 1.0 - 1e-12:
            continue                                   # half-plane: both legs share one edge
        pieces.append((n, float(n @ cone.apex)))
    return pieces


def _shift_cone(cone: VOCone, delta: float) -> VOCone:
    """The cone with both edges pulled inward by delta [m/s] (apex moved
    delta / sin(half-angle) along its axis): it forbids only velocities more
    than delta inside the original cone."""
    s = math.sin(cone.half_angle)
    return VOCone(apex=cone.apex + (delta / max(s, 1e-9)) * cone.axis,
                  axis=cone.axis, half_angle=cone.half_angle)


def _min_shift(R, u, budget, cones, sb_units, epsilon):
    """Closed-form smallest common shift delta* that frees a heading, or None.

    On the ring v(theta) = R (cos theta, sin theta) every cone edge gives the
    sinusoid f(theta) = n . v(theta) - c = R cos(theta - phi) - c (phi =
    direction of n), all with the same amplitude R.  The worst shortfall is
    g(theta) = max_cones min_edges f(theta); delta* = min g over the allowed
    arcs (course budget, starboard constraints).  The minimum lies at an arc
    end, at the bottom of a piece (theta = phi + pi) or where two pieces
    cross:  sin(theta - (phi1 + phi2)/2) = (c2 - c1) / (2 R sin((phi2 - phi1)/2)).
    None if the budget and starboard constraints allow no heading at all.
    """
    tol = 1e-7 * max(1.0, R)
    cone_pieces = [_cone_edge_pieces(cn) for cn in cones]
    flat = [pc for cp in cone_pieces for pc in cp]

    thetas: list[float] = []
    # (1) arc ends: starboard boundaries and budget edges meeting the ring
    for p, d, ray in _boundary_prims([], sb_units) + _budget_rays(u, budget):
        thetas.extend(math.atan2(float(q[1]), float(q[0])) for q in _prim_circle(p, d, ray, R))
    # (2) bottoms of the pieces
    phis = [math.atan2(float(n[1]), float(n[0])) for n, _ in flat]
    thetas.extend(ph + math.pi for ph in phis)
    # (3) pairwise crossings (equal amplitude -> one asin each)
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            s = math.sin(0.5 * (phis[j] - phis[i]))
            if abs(s) < 1e-12:
                continue
            rhs = (flat[j][1] - flat[i][1]) / (2.0 * R * s)
            if abs(rhs) > 1.0:
                continue
            mid = 0.5 * (phis[i] + phis[j])
            a = math.asin(rhs)
            thetas.extend((mid + a, mid + math.pi - a))
    if not thetas:
        return None

    th    = np.array(thetas)
    cands = R * np.column_stack((np.cos(th), np.sin(th)))
    ok  = _feasible_mask(cands, [], sb_units, tol, epsilon)
    ok &= (cands @ u) >= math.cos(budget) * R - tol
    if not ok.any():
        return None
    fc = cands[ok]
    g  = np.zeros(len(fc))
    for cp in cone_pieces:
        g = np.maximum(g, np.min(np.stack([fc @ n - c for n, c in cp]), axis=0))
    return float(g.min())


def _min_shift_bisect(pick, cones: list[VOCone], R: float, iters: int):
    """Smallest shift for which pick(shifted cones) succeeds, by bisection --
    numerical safety only, for exact ties at the closed-form delta*.
    Returns (v, delta), or None."""
    if not cones:
        return None
    hi = R + max(math.hypot(float(cn.apex[0]), float(cn.apex[1])) for cn in cones) + 1e-3
    v_hi = pick([_shift_cone(cn, hi) for cn in cones])
    if v_hi is None:
        return None
    lo = 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        v_mid = pick([_shift_cone(cn, mid) for cn in cones])
        if v_mid is None:
            lo = mid
        else:
            hi, v_hi = mid, v_mid
    return v_hi, hi


# ---------------------------------------------------------------------------
# Achievable passing distance and velocity selection
# ---------------------------------------------------------------------------

def achievable_pass_distance(
    apex:     np.ndarray,
    axis:     np.ndarray,
    dist:     float,
    R:        float,
    u:        np.ndarray,
    budget:   float,
    sb_unit:  tuple[np.ndarray, np.ndarray] | None = None,
    epsilon:  float = 1e-6,
) -> float | None:
    """Largest passing distance D* to one obstacle reachable on the speed ring.

    Heading v on |v| = R, within `budget` of u and (if sb_unit is given)
    satisfying that obstacle's starboard constraint, passes the obstacle at
    dist * sin(phi), phi = angle(v - apex, axis); phi >= 90 deg means the
    range does not close, i.e. the pass is at the current distance.  So
    D* = dist * sin(min(phi_max, 90 deg)).  phi_max over an admissible arc
    lies at an arc end (budget edge, starboard boundary), at a tangent
    point from the apex to the ring, or reaches 90 deg on the line through
    the apex perpendicular to the axis; all are evaluated in closed form.
    Returns None if no heading is admissible.  D* <= dist always.
    """
    tol  = 1e-7 * max(1.0, R)
    sbs  = [sb_unit] if sb_unit is not None else []
    pts: list[np.ndarray] = [R * d for _, d, _ in _budget_rays(u, budget)]
    na = math.hypot(float(apex[0]), float(apex[1]))
    if na > R:
        base, half = math.atan2(float(apex[1]), float(apex[0])), math.acos(R / na)
        pts.extend(R * np.array([math.cos(base + s * half), math.sin(base + s * half)]) for s in (-1.0, 1.0))
    pts.extend(_prim_circle(apex, np.array([-axis[1], axis[0]]), False, R))
    for p, d, ray in _boundary_prims([], sbs):
        pts.extend(_prim_circle(p, d, ray, R))
    if not pts:
        return None
    cands = np.array(pts)
    ok  = _feasible_mask(cands, [], sbs, tol, epsilon)
    ok &= (cands @ u) >= math.cos(budget) * R - tol
    if not ok.any():
        return None
    rel  = cands[ok] - apex
    nrm  = np.sqrt((rel * rel).sum(axis=1))
    cosp = np.where(nrm > 1e-9, (rel @ axis) / np.maximum(nrm, 1e-9), 0.0)
    c    = float(cosp.min())
    if c <= 0.0:
        return dist
    return dist * math.sqrt(max(0.0, 1.0 - c * c))


def solve_vo_achievable(
    v_pref:            np.ndarray,
    obstacles:         list[tuple[np.ndarray, np.ndarray, float, float, tuple | None]],
    nominal_unit:      np.ndarray,
    course_budget:     float,
    achievable_ratio:  float,
    v_prev:            np.ndarray | None = None,
    continuity_weight: float = 0.0,
    epsilon:           float = 1e-6,
    bisect_iters:      int   = 20,
) -> tuple[np.ndarray, list[float], float, int]:
    """Velocity selection.  Always at the preferred speed R = |v_pref|;
    u = nominal_unit, alpha = achievable_ratio.

    obstacles: (apex, unit axis, dist, D_j, starboard unit (axis, ref) or None)
    per obstacle.

      1. Normal case.  D*_j = best passing distance to j alone on the ring
         within the course budget (and j's starboard constraint),
         achievable_pass_distance.  Radius r_j = min(D_j, alpha * D*_j) --
         never above alpha * dist, so there is no inside-D half-plane.  Pick
         (Eq. 19): min |angle(v, v_pref)| + lambda |angle(v, v_prev)| on the
         ring, within the budget and starboard constraints, outside all cones.
         Always feasible for a single obstacle.
      2. Conflict (several obstacles leave no heading): every cone is pulled
         in parallel by the same velocity shift delta [m/s] (_shift_cone).
         This lowers the passing distance to obstacle j by about
         delta * t_cpa_j: one number reduces all passing distances, more for
         encounters further ahead in time, which will be re-planned before
         they happen.  delta* = the smallest shift that frees a heading
         (closed form, _min_shift); delta*/alpha is applied for the same
         slack as the radii, then the Eq.-19 pick.  delta starts at 0 when a
         conflict appears, so the transition is continuous.
      3. Only if the starboard constraints exclude each other within the
         budget (no shift helps) does the vessel keep its previous course.

    Returns (v, radii, delta, case) -- delta = shift used (0 in case 1).
    """
    R = math.hypot(float(v_pref[0]), float(v_pref[1]))
    u = np.asarray(nominal_unit, dtype=float)
    if R < 1e-9 or not obstacles:
        return np.asarray(v_pref, dtype=float).copy(), [], 0.0, 1
    has_prev = v_prev is not None and float(v_prev[0] ** 2 + v_prev[1] ** 2) > 1e-18
    prev = v_prev if has_prev else None
    lam  = continuity_weight

    sb_units: list[tuple[np.ndarray, np.ndarray]] = []
    cones:    list[VOCone] = []
    radii:    list[float]  = []
    for apex, axis, dist, D, sb in obstacles:
        ds = achievable_pass_distance(apex, axis, dist, R, u, course_budget, sb, epsilon)
        r  = D if ds is None else min(D, achievable_ratio * ds)
        cones.append(VOCone(apex=apex, axis=axis, half_angle=math.asin(min(r / dist, 1.0))))
        radii.append(r)
        if sb is not None:
            sb_units.append(sb)

    def pick(cs):
        return _ring_pick(v_pref, prev, lam, R, u, course_budget, cs, sb_units, epsilon)

    # --- 1. Normal case ----------------------------------------------------
    v = pick(cones)
    if v is not None:
        return v, radii, 0.0, 1

    # --- 2. Conflict: shift all cones by the same amount --------------------
    delta = _min_shift(R, u, course_budget, cones, sb_units, epsilon)
    if delta is not None:
        shift = max(delta, 0.0) / achievable_ratio + 1e-9 * max(1.0, R)
        v = pick([_shift_cone(cn, shift) for cn in cones])
        if v is not None:
            return v, radii, shift, 2
        res = _min_shift_bisect(pick, cones, R, bisect_iters)
        if res is not None:
            return res[0], radii, res[1], 2

    # --- 3. Starboard constraints exclude each other: previous course -----
    head = np.asarray(v_prev if has_prev else v_pref, dtype=float)
    return head * (R / math.hypot(float(head[0]), float(head[1]))), radii, float("nan"), 3
