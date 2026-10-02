"""policies/vo_analytic.py

Analytic Velocity Obstacle policy for maritime collision avoidance with
COLREGs support -- the low-level layer of the interaction-aware predictor.

Based on:
    Kuwata et al. (2011) "Safe Maritime Navigation with COLREGS Using
    Velocity Obstacles", IEEE/RSJ IROS.

Method ("achievable passing distance")
--------------------------------------
Each step, for the own vessel, with R = preferred speed and u = nominal
heading:

1.  Per obstacle j (COLREGs role from the interaction graph): the (R)VO cone
    geometry -- apex = v_j (RVO midpoint for HEAD_ON), axis = line of sight
    -- plus a starboard constraint for GIVE_WAY / HEAD_ON.
2.  D*_j = the largest passing distance to j that any heading on the speed
    ring |v| = R within the course budget (and j's starboard constraint)
    can achieve -- closed form, achievable_pass_distance.
3.  Cone radius r_j = min(D_j, alpha * D*_j): the full passing distance when
    it is reachable, otherwise alpha of the best reachable one.
4.  Pick (paper Eq. 19): the heading on the speed ring, within the course
    budget and the starboard constraints and outside all cones, minimising
    |angle(v, v_pref)| + lambda * |angle(v, v_prev)|.
5.  If several obstacles together leave no heading, all cones are pulled
    in parallel by the same velocity shift delta, which lowers the passing
    distance to each obstacle by about delta * t_cpa -- closed form, the
    smallest shift that frees a heading, divided by alpha -- then step 4.

The speed is never changed, so every response is a course change (what the
responding filter scores).  All radii vary continuously with the geometry,
also inside D (r_j <= alpha * distance, so the cone never degenerates to
the "do not close" half-plane that makes a vessel escort another).

COLREGs role behaviour
----------------------
GIVE_WAY        : passing distance d_pass, starboard constraint
STAND_ON        : cone at d_coll -- the last-moment action of Rule 17(b).
                  Paper Eq. 12 has no constraint here; non-responsive
                  vessels (which hold STAND_ON) get no obstacles at all in
                  the rollout, see prediction/vo_rollout.py
HEAD_ON         : d_pass, RVO apex (Rule 14), starboard constraint
OVERTAKING      : d_pass, no side constraint
AVOID_COLLISION : d_pass, no side constraint

When no ColregsGraph is supplied all obstacles default to AVOID_COLLISION.

Notes
-----
- NED frame throughout; velocities are (vn, ve), angles clockwise positive.
- Returns CVAction(vn, ve) -- compatible with CVAdapter.
"""

from __future__ import annotations

import math

import numpy as np

from src.vessel.state import VesselState, CVAction
from src.vessel.base import BasePolicy
from src.colregs_graph.colregs_graph import VesselRole, ColregsGraph
from src.policies.debug_types import VODebugFrame, VOConeDebugData, StarboardConstraintDebugData

from .vo_common import (
    VOCone,
    RoleConfig,
    RolePassDistances,
    role_config,
)


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


# ---------------------------------------------------------------------------
# VOPolicyAnalytic
# ---------------------------------------------------------------------------

class VOPolicyAnalytic(BasePolicy):
    """Analytic Velocity Obstacle policy with COLREGs role-based constraints
    (see the module docstring for the method).

    Returns CVAction(vn, ve) -- compatible with CVAdapter.

    Build it with prediction.vo_rollout.make_vo_policy(VOAgentConfig), which
    holds the parameter defaults.

    Parameters
    ----------
    nominal_psi : float
        Nominal heading [rad], 0 = north, clockwise positive (NED).  The
        course budget is centred on it.
    ref_speed : float
        Preferred speed [m/s] (the speed ring).
    tcpa_threshold : float
        Obstacles without a COLREGs-graph edge are ignored when their TCPA
        >= tcpa_threshold [s].  Vessels with an edge are always included.
    d_pass : float
        Desired passing distance D [m].
    d_coll : float
        Passing distance kept from STAND_ON obstacles [m].  Must be < d_pass.
    los_lookahead : float
        LOS guidance lookahead distance [m] (paper Eq. 15).
    head_on_rvo_weight : float
        RVO apex weight for HEAD_ON cones.  0.5 = full RVO (equal shared
        responsibility, Rule 14).
    d_pass_roles : RolePassDistances | None
        Per-role passing distances; None -> d_pass for every role, d_coll for
        STAND_ON.
    colregs_graph : ColregsGraph | None
        Pairwise COLREGs role graph.  None -> all obstacles AVOID_COLLISION.
    own_vessel_idx : int
        Index of this vessel in the graph (0 = ego).
    other_vessel_indices : list[int] | None
        Graph indices of the other vessels, in the order of other_states
        passed to select_action.  None -> sequential 0, 1, 2, ...
    origin_pos : ndarray | None
        LOS track-line origin [north, east].  None -> the position at the
        first select_action() call.
    continuity_weight : float
        lambda in Eq. 19: weight of the deviation from the previously
        selected heading, relative to the deviation from the preferred
        heading (weight 1).  Stops the pick flipping between near-tied
        headings from step to step.
    course_budget_deg : float
        Largest course alteration from the nominal heading [deg]; 180 = no
        limit.  About 60 is recommended: much larger budgets admit headings
        far off course that barely close the range, and the pick can jump
        between those and headings near the preferred course.
    achievable_ratio : float
        alpha in (0, 1].  Fraction of the best reachable passing distance
        demanded when D itself is out of reach (and of the conflict shift's
        slack).  < 1 leaves room so the pick can stay nearer the preferred
        course; 1 = maximum effort.  Keep slightly below 1 (e.g. 0.99): at
        exactly 1 the feasible set is a single heading and floating-point
        rounding decides whether it is found.

    Class attributes
    ----------------
    emit_debug : bool
        True populates last_debug after each select_action() call.
    """

    emit_debug: bool = True

    def __init__(
        self,
        *,
        nominal_psi:          float,
        ref_speed:            float,
        tcpa_threshold:       float,
        d_pass:               float,
        d_coll:               float,
        los_lookahead:        float,
        head_on_rvo_weight:   float,
        continuity_weight:    float,
        course_budget_deg:    float,
        achievable_ratio:     float,
        d_pass_roles:         RolePassDistances | None = None,
        origin_pos:           np.ndarray | None        = None,
        colregs_graph:        ColregsGraph | None       = None,
        own_vessel_idx:       int                       = 0,
        other_vessel_indices: list[int] | None          = None,
    ) -> None:
        if d_coll >= d_pass:
            raise ValueError(
                f"d_coll ({d_coll}) must be strictly less than d_pass ({d_pass})"
            )
        if not 0.0 <= head_on_rvo_weight <= 1.0:
            raise ValueError(
                f"head_on_rvo_weight must be in [0, 1], got {head_on_rvo_weight}"
            )
        if not 0.0 < course_budget_deg <= 180.0:
            raise ValueError(
                f"course_budget_deg must be in (0, 180], got {course_budget_deg}"
            )
        if not 0.0 < achievable_ratio <= 1.0:
            raise ValueError(f"achievable_ratio must be in (0, 1], got {achievable_ratio}")
        if continuity_weight < 0.0:
            raise ValueError(
                f"continuity_weight must be >= 0.0, got {continuity_weight}"
            )

        self.nominal_psi        = nominal_psi
        self.ref_speed          = ref_speed
        self.tcpa_threshold     = tcpa_threshold
        self.d_pass             = d_pass
        self.d_coll             = d_coll
        self._d_pass_r          = (
            d_pass_roles if d_pass_roles is not None
            else RolePassDistances.from_scalars(d_pass, d_coll)
        )
        self.los_lookahead      = los_lookahead
        self.head_on_rvo_weight = head_on_rvo_weight
        self.continuity_weight  = continuity_weight
        self.course_budget      = math.radians(course_budget_deg)
        self.achievable_ratio   = achievable_ratio
        self._v_prev: np.ndarray | None = None

        self._v_pref      = np.array([
            ref_speed * np.cos(nominal_psi),
            ref_speed * np.sin(nominal_psi),
        ])
        self._v_pref_unit = self._v_pref / np.linalg.norm(self._v_pref)

        self._origin_pos: np.ndarray | None = (
            origin_pos.copy() if origin_pos is not None else None
        )
        self.last_debug:  VODebugFrame | None = None

        self._colregs_graph        = colregs_graph
        self._own_vessel_idx       = own_vessel_idx
        self._other_vessel_indices = other_vessel_indices
        self._role_map_cache: dict | None = None

    def set_colregs_graph(self, graph: ColregsGraph | None) -> None:
        """Replace the COLREGs graph (e.g. from a live classifier)."""
        self._colregs_graph  = graph
        self._role_map_cache = None

    # ------------------------------------------------------------------
    # COLREGs role annotation
    # ------------------------------------------------------------------

    def _annotate_roles(
        self,
        other_states: list[VesselState],
    ) -> list[tuple[VesselState, VesselRole, bool]]:
        """Attach a VesselRole and graph-edge flag to each other vessel.

        The bool is True when the vessel has an edge in the COLREGs graph,
        which is used to bypass the TCPA pre-filter.

        With a graph, vessels without an edge are dropped entirely: vessels
        that i does not interact with impose no constraint (paper Eq. 13).
        Without a graph, every vessel is AVOID_COLLISION and goes through the
        TCPA pre-filter (plain VO behaviour).
        """
        if self._colregs_graph is None:
            return [(s, VesselRole.AVOID_COLLISION, False) for s in other_states]

        if self._role_map_cache is None:
            self._role_map_cache = dict(self._colregs_graph.get_roles(self._own_vessel_idx))
        role_map = self._role_map_cache
        indices  = (
            self._other_vessel_indices
            if self._other_vessel_indices is not None
            else list(range(len(other_states)))
        )
        return [
            (obs, role_map[idx], True)
            for obs, idx in zip(other_states, indices)
            if idx in role_map
        ]

    # ------------------------------------------------------------------
    # BasePolicy interface
    # ------------------------------------------------------------------

    def select_action(
            self,
            own_state:    VesselState,
            other_states: list[VesselState],
        ) -> CVAction:
            """Select a velocity action, resolving COLREGs roles from the stored graph."""
            annotated = self._annotate_roles(other_states)
            own_pos   = own_state.position
            own_vel   = np.array([own_state.vn, own_state.ve])

            # Initialise track line origin on first call
            if self._origin_pos is None:
                self._origin_pos = own_pos.copy()

            # --- LOS guidance (paper Eqs. 14-16) ---------------------------------
            _pu1, _pu0 = float(self._v_pref_unit[1]), float(self._v_pref_unit[0])
            _dx = own_pos[0] - self._origin_pos[0]
            _dy = own_pos[1] - self._origin_pos[1]
            e   = _dx * _pu1 - _dy * _pu0
            psi_d      = self.nominal_psi + math.atan2(e, self.los_lookahead)
            _cn, _sn   = math.cos(psi_d), math.sin(psi_d)
            v_pref_now = np.array([self.ref_speed * _cn, self.ref_speed * _sn])

            # --- 1. Obstacles that constrain this vessel ----------------------
            # Vessels with a COLREGs graph edge are always included; others
            # only while their TCPA is below tcpa_threshold.
            _own_vn, _own_ve = own_vel[0], own_vel[1]
            _own_n,  _own_e  = own_state.north, own_state.east
            threats: list[tuple[VesselState, VesselRole]] = []
            for obs, role, in_graph in annotated:
                if in_graph:
                    threats.append((obs, role))
                    continue
                _rvn = _own_vn - obs.vn
                _rve = _own_ve - obs.ve
                _rsq = _rvn * _rvn + _rve * _rve
                if _rsq < 1e-12:
                    threats.append((obs, role))
                    continue
                _rpn = obs.north - _own_n
                _rpe = obs.east  - _own_e
                if max((_rpn * _rvn + _rpe * _rve) / _rsq, 0.0) < self.tcpa_threshold:
                    threats.append((obs, role))

            if not threats:
                if self.emit_debug:
                    self.last_debug = VODebugFrame(
                        v_pref_vn=float(v_pref_now[0]), v_pref_ve=float(v_pref_now[1]),
                        v_sel_vn=float(v_pref_now[0]),  v_sel_ve=float(v_pref_now[1]),
                        cones=[], ref_speed=self.ref_speed,
                        fallback_active=False,
                    )
                self._v_prev = v_pref_now.copy()
                return CVAction(vn=float(v_pref_now[0]), ve=float(v_pref_now[1]))

            configs: list[RoleConfig] = [
                role_config(role, self._d_pass_r, self.d_coll)
                for _, role in threats
            ]

            # --- 2. Starboard reference velocities ----------------------------
            # For HEAD_ON obstacles the starboard constraint is anchored to a
            # laterally-shifted reference velocity: rvo_weight * (lateral
            # component of v_own - v_obs), consistent with the RVO cone apex.
            def _sb_ref_vel(obs: VesselState, role: VesselRole) -> np.ndarray:
                obs_vel_arr = np.array([obs.vn, obs.ve])
                if role is not VesselRole.HEAD_ON:
                    return obs_vel_arr
                bearing       = obs.position - own_pos
                bearing       = bearing / np.linalg.norm(bearing)
                lateral       = np.array([-bearing[1], bearing[0]])
                rvo_shift     = self.head_on_rvo_weight * (own_vel - obs_vel_arr)
                lateral_shift = np.dot(rvo_shift, lateral) * lateral
                return obs_vel_arr + lateral_shift

            # --- 3. Per-obstacle geometry -------------------------------------
            obstacles = []
            if self.emit_debug:
                debug_starboard: list[StarboardConstraintDebugData] = []
            for (obs, role), cfg in zip(threats, configs):
                rel  = obs.position - own_pos
                dist = float(np.linalg.norm(rel))
                if dist < 1e-6:
                    continue
                axis    = rel / dist
                obs_vel = np.array([obs.vn, obs.ve])
                apex    = (obs_vel + self.head_on_rvo_weight * (own_vel - obs_vel)
                           if role is VesselRole.HEAD_ON else obs_vel)
                sb_unit = None
                if cfg.starboard_only:
                    sb_ref  = _sb_ref_vel(obs, role)
                    sb_unit = (axis, sb_ref)
                    if self.emit_debug:
                        debug_starboard.append(StarboardConstraintDebugData(
                            apex_vn=float(sb_ref[0]), apex_ve=float(sb_ref[1]),
                            bearing_n=float(axis[0]), bearing_e=float(axis[1]),
                        ))
                obstacles.append((apex, axis, dist, cfg.primary_r, sb_unit))

            # --- 4. Velocity selection ----------------------------------------
            v_sel, radii, _, case = solve_vo_achievable(
                v_pref            = v_pref_now,
                obstacles         = obstacles,
                nominal_unit      = self._v_pref_unit,
                course_budget     = self.course_budget,
                achievable_ratio  = self.achievable_ratio,
                v_prev            = self._v_prev,
                continuity_weight = self.continuity_weight,
            )

            if self.emit_debug:
                self.last_debug = VODebugFrame(
                    v_pref_vn=float(v_pref_now[0]), v_pref_ve=float(v_pref_now[1]),
                    v_sel_vn=float(v_sel[0]),       v_sel_ve=float(v_sel[1]),
                    cones=[
                        VOConeDebugData(
                            apex_vn=apex[0], apex_ve=apex[1],
                            axis_n=axis[0],  axis_e=axis[1],
                            half_angle=math.asin(min(r / dist, 1.0)),
                        )
                        for (apex, axis, dist, _, _), r in zip(obstacles, radii)
                    ],
                    ref_speed=self.ref_speed,
                    fallback_active=case > 1,
                    starboard_constraints=debug_starboard,
                    forward_unit_n=float(self._v_pref_unit[0]),
                    forward_unit_e=float(self._v_pref_unit[1]),
                )
            self._v_prev = v_sel.copy()
            return CVAction(vn=float(v_sel[0]), ve=float(v_sel[1]))


class NominalLOSPolicy(BasePolicy):
    """Follow the nominal mission track via LOS guidance, ignoring all traffic.

    Wraps a VOPolicyAnalytic instance but always calls it with an empty
    obstacle list, so no VO cone / COLREGs reasoning ever engages — the
    vessel holds its nominal course/speed regardless of nearby traffic.
    Used for non-responsive vessels: COLREGs' stand-on rule still lets the
    stand-on vessel react to imminent danger, which is not what "does not
    respond" should mean in simulation, so this bypasses that reaction
    entirely instead of relying on the STAND_ON role's d_coll cone.
    """

    def __init__(self, inner: VOPolicyAnalytic) -> None:
        self._inner = inner

    def select_action(
        self,
        own_state:     VesselState,
        _other_states: list[VesselState],
    ) -> CVAction:
        return self._inner.select_action(own_state, [])

    @property
    def last_debug(self) -> VODebugFrame | None:
        return self._inner.last_debug
