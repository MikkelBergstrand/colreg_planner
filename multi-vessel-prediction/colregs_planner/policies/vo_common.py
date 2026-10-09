"""policies/vo_common.py

Shared primitives for Velocity Obstacle collision avoidance policies.

Imported by both VOPolicyKuwata (grid search) and VOPolicyAnalytic (analytic
projection).  Neither policy file should duplicate any of the geometry,
role configuration, or masking logic defined here.

Contents
--------
tcpa                        Time to CPA pre-filter
VOCone                      VO cone geometry + containment tests
build_vo_cone               Construct a VOCone from vessel states
project_onto_cone_boundary  Project a velocity onto a single cone exterior
project_through_cones       Sequential projection through a list of cones
RoleConfig                  Per-obstacle VO configuration
RolePassDistances           Role-dependent passing distances
role_config                 Map VesselRole -> RoleConfig
starboard_mask              Per-obstacle starboard candidate mask
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from colregs_planner.colregs_graph.colregs_graph import VesselRole


# ---------------------------------------------------------------------------
# CPA pre-filter
# ---------------------------------------------------------------------------

def tcpa(
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
) -> float:
    """Time to closest point of approach [s], clamped to 0 if diverging.

    Parameters
    ----------
    rel_pos : p_obs - p_own  [m]
    rel_vel : v_own - v_obs  [m/s]

    Derivation: relative position of obs w.r.t. own is
        r(t) = rel_pos - t * rel_vel
    Minimising |r(t)|^2 gives TCPA = dot(rel_pos, rel_vel) / |rel_vel|^2.
    """
    rsq = rel_vel[0]*rel_vel[0] + rel_vel[1]*rel_vel[1]
    if rsq < 1e-12:
        return 0.0
    t = (rel_pos[0]*rel_vel[0] + rel_pos[1]*rel_vel[1]) / rsq
    return t if t > 0.0 else 0.0


# ---------------------------------------------------------------------------
# VO cone
# ---------------------------------------------------------------------------

def _rotate2d(v: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 2-D NED vector by angle [rad], positive = clockwise."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


@dataclass
class VOCone:
    """Velocity Obstacle cone for one obstacle.

    Attributes
    ----------
    apex       : cone apex = v_obs [m/s] (or RVO-shifted apex for HEAD_ON)
    axis       : unit vector from own vessel toward obstacle (centreline)
    half_angle : half opening angle [rad]
    left_leg   : unit vector of left (port-side) cone boundary
    right_leg  : unit vector of right (starboard-side) cone boundary

    Notes
    -----
    NED frame: positive rotation = clockwise.
    Left leg is to port  (counter-clockwise = negative angle in NED).
    Right leg is to starboard (clockwise = positive angle in NED).
    """
    apex:           np.ndarray
    axis:           np.ndarray
    half_angle:     float
    cos_half_angle: float       = field(init=False)
    left_leg:       np.ndarray  = field(init=False)
    right_leg:      np.ndarray  = field(init=False)

    def __post_init__(self) -> None:
        self.cos_half_angle = np.cos(self.half_angle)
        self.left_leg  = _rotate2d(self.axis, -self.half_angle)
        self.right_leg = _rotate2d(self.axis, +self.half_angle)

    def contains(self, v: np.ndarray) -> bool:
        """Return True if velocity v falls inside this VO cone."""
        rel = v - self.apex
        dot = float(np.dot(rel, self.axis))
        if dot <= 0.0:
            return False
        return dot * dot > self.cos_half_angle ** 2 * float(np.dot(rel, rel))

    def contains_batch(self, candidates: np.ndarray) -> np.ndarray:
        """Vectorized containment check for an (N, 2) candidate array.

        Returns a (N,) bool array; True where the candidate falls inside
        this cone.  Equivalent to [self.contains(v) for v in candidates]
        but avoids the Python loop.
        """
        rel = candidates - self.apex[np.newaxis, :]       # (N, 2)
        dot = rel @ self.axis                              # (N,)
        sq  = np.einsum("ij,ij->i", rel, rel)             # (N,)
        return (dot > 0) & (dot ** 2 > self.cos_half_angle ** 2 * sq)

def build_vo_cone(
    own_pos:    np.ndarray,
    obs_pos:    np.ndarray,
    obs_vel:    np.ndarray,
    combined_r: float,
    own_vel:    np.ndarray | None = None,
    rvo_weight: float = 0.0,
) -> VOCone:
    """Construct the VO (or RVO) cone for one obstacle.

    Parameters
    ----------
    own_pos    : own vessel position [m]
    obs_pos    : obstacle position [m]
    obs_vel    : obstacle velocity [m/s]
    combined_r : Minkowski sum radius (d_pass or d_coll) [m]
    own_vel    : own vessel velocity [m/s].  Required when rvo_weight > 0.
    rvo_weight : interpolation weight in [0, 1] between plain VO and full
                 RVO.  0.0 places the apex at v_obs (standard VO).  0.5
                 places it at (v_own + v_obs) / 2 (full RVO), encoding the
                 assumption that the obstacle takes equal responsibility for
                 avoidance.  Only pass a non-zero value for HEAD_ON primary
                 cones; fallback cones should always use 0.0 (conservative).

    Inside combined_r the cone degenerates to the half-plane of
    range-closing velocities.

    Notes
    -----
    RVO apex interpolation:
        apex = v_obs + rvo_weight * (v_own - v_obs)
             = (1 - rvo_weight) * v_obs + rvo_weight * v_own

    At rvo_weight = 0.5 this gives apex = (v_own + v_obs) / 2, the classic
    RVO formulation of van den Berg et al. (2008).  The cone geometry (axis,
    half_angle) is unchanged; only the apex shifts, which moves the entire
    forbidden region in velocity space toward v_own and encodes the
    anticipated cooperative manoeuvre of the obstacle.
    """
    if rvo_weight != 0.0 and own_vel is None:
        raise ValueError("own_vel must be provided when rvo_weight > 0")

    rel_pos = obs_pos - own_pos
    dist    = np.linalg.norm(rel_pos)

    apex = obs_vel.copy() if (rvo_weight == 0.0 or own_vel is None) else (
        obs_vel + rvo_weight * (own_vel - obs_vel)
    )

    r = combined_r

    if dist < r:
        return VOCone(
            apex=apex,
            axis=rel_pos / (dist + 1e-9),
            half_angle=np.pi / 2.0,
        )

    return VOCone(
        apex=apex,
        axis=rel_pos / dist,
        half_angle=np.arcsin(np.clip(r / dist, -1.0, 1.0)),
    )


# ---------------------------------------------------------------------------
# Cone projection helpers
# ---------------------------------------------------------------------------

def project_onto_cone_boundary(
    v:    np.ndarray,
    cone: VOCone,
) -> np.ndarray:
    """Project v onto the exterior of a VO cone.

    If v is already outside the cone it is returned unchanged.  If v is
    inside the cone, the nearest point on the cone boundary is returned.

    Used in fallback mode (grid policy) to project v_pref_now onto d_pass
    cone boundaries so the deviation cost pulls d_coll-admissible candidates
    toward the nearest d_pass-compliant velocity rather than the blocked
    nominal heading.

    Parameters
    ----------
    v    : velocity to project [m/s], shape (2,)
    cone : VOCone to project against

    Returns
    -------
    v_proj : projected velocity [m/s], shape (2,)
    """
    rel      = v - cone.apex
    rel_norm = np.linalg.norm(rel)

    # v coincides with apex — step out along an arbitrary boundary ray
    if rel_norm < 1e-9:
        perp = np.array([-cone.axis[1], cone.axis[0]])
        return cone.apex + rel_norm * (
            np.cos(cone.half_angle) * cone.axis
            + np.sin(cone.half_angle) * perp
        )

    # Already outside — no projection needed
    cos_a = np.dot(rel, cone.axis) / rel_norm
    if cos_a <= np.cos(cone.half_angle):
        return v

    # Decompose into axial + perpendicular, clamp to half_angle
    axial     = np.dot(rel, cone.axis) * cone.axis
    perp_vec  = rel - axial
    perp_norm = np.linalg.norm(perp_vec)

    if perp_norm < 1e-9:
        # v on cone axis — pick arbitrary boundary direction
        perp_unit = np.array([-cone.axis[1], cone.axis[0]])
    else:
        perp_unit = perp_vec / perp_norm

    return cone.apex + rel_norm * (
        np.cos(cone.half_angle) * cone.axis
        + np.sin(cone.half_angle) * perp_unit
    )


def project_through_cones(
    v:     np.ndarray,
    cones: list[VOCone],
) -> np.ndarray:
    """Sequentially project v out of each cone in the list.

    Used by the grid policy for its fallback anchor.

    Projection is sequential: the result of projecting out of cone i is used
    as input to cone i+1.  This is not globally optimal but is fast and
    produces sensible results for typical maritime encounter geometries where
    cones overlap moderately.
    """
    v_proj = v.copy()
    for cone in cones:
        v_proj = project_onto_cone_boundary(v_proj, cone)
    return v_proj


# ---------------------------------------------------------------------------
# Role configuration
# ---------------------------------------------------------------------------

@dataclass
class RoleConfig:
    """Per-obstacle VO configuration derived from COLREGs role.

    Attributes
    ----------
    primary_r      : cone radius for primary admissibility pass [m]
    has_fallback   : whether to attempt fallback with d_coll before full stop
    starboard_only : whether to mask / filter port-side candidates

    COLREGs role behaviour (primary_r comes from RolePassDistances)
    ----------------------------------------------------------------
    GIVE_WAY        : primary d_pass_r.give_way,        fallback d_coll, starboard candidates only
    STAND_ON        : primary d_pass_r.stand_on,        no fallback (full stop on failure)
    HEAD_ON         : primary d_pass_r.head_on,         fallback d_coll, starboard candidates only
    OVERTAKING      : primary d_pass_r.overtaking,      fallback d_coll, no side constraint
    AVOID_COLLISION : primary d_pass_r.avoid_collision, fallback d_coll, no side constraint
    """
    primary_r:      float
    has_fallback:   bool
    starboard_only: bool


@dataclass
class RolePassDistances:
    """Role-dependent passing distance D^i_R [m].

    Give-way vessels empirically target a larger clearance than other roles,
    motivating a per-role rather than a uniform passing distance. STAND_ON
    defaults to d_coll (the collision-avoidance radius).
    """
    give_way:        float
    head_on:         float
    overtaking:      float
    avoid_collision: float
    stand_on:        float  # primary radius for stand-on; typically equals d_coll

    @classmethod
    def from_scalars(cls, d_pass: float, d_coll: float) -> "RolePassDistances":
        """Construct uniform role distances: d_pass for active roles, d_coll for stand-on."""
        return cls(
            give_way=d_pass, head_on=d_pass,
            overtaking=d_pass, avoid_collision=d_pass,
            stand_on=d_coll,
        )

    @classmethod
    def from_dict(cls, raw: dict, d_pass: float, d_coll: float) -> "RolePassDistances":
        """Per-role overrides from a YAML block; missing roles fall back to
        d_pass (d_coll for stand-on)."""
        unknown = set(raw) - {"give_way", "head_on", "overtaking", "avoid_collision", "stand_on"}
        if unknown:
            raise ValueError(f"Unknown d_pass_roles key(s): {sorted(unknown)}")
        return cls(
            give_way        = float(raw.get("give_way",        d_pass)),
            head_on         = float(raw.get("head_on",         d_pass)),
            overtaking      = float(raw.get("overtaking",      d_pass)),
            avoid_collision = float(raw.get("avoid_collision", d_pass)),
            stand_on        = float(raw.get("stand_on",        d_coll)),
        )


def role_config(role: VesselRole, d_pass_r: RolePassDistances, d_coll: float) -> RoleConfig:
    """Map a COLREGs VesselRole to its VO RoleConfig."""
    match role:
        case VesselRole.GIVE_WAY:
            return RoleConfig(primary_r=d_pass_r.give_way,        has_fallback=True,  starboard_only=True)
        case VesselRole.STAND_ON:
            return RoleConfig(primary_r=d_pass_r.stand_on,        has_fallback=False, starboard_only=False)
        case VesselRole.HEAD_ON:
            return RoleConfig(primary_r=d_pass_r.head_on,         has_fallback=True,  starboard_only=True)
        case VesselRole.OVERTAKING:
            return RoleConfig(primary_r=d_pass_r.overtaking,      has_fallback=True,  starboard_only=False)
        case VesselRole.AVOID_COLLISION:
            return RoleConfig(primary_r=d_pass_r.avoid_collision,  has_fallback=True,  starboard_only=False)
        case _:
            raise ValueError(f"Unknown VesselRole: {role}")


# ---------------------------------------------------------------------------
# Starboard mask (grid policy — vectorised over candidate array)
# ---------------------------------------------------------------------------

def starboard_mask(
    candidates: np.ndarray,
    own_pos:    np.ndarray,
    obs_pos:    np.ndarray,
    obs_vel:    np.ndarray,
) -> np.ndarray:
    """Return a boolean mask that is True for candidates that pass the
    obstacle on the starboard (right) side of own vessel.

    A candidate velocity v produces a starboard pass when the relative
    velocity (v - v_obs) crosses to the right of the bearing vector to the
    obstacle.  In 2D NED (north = axis 0, east = axis 1, clockwise positive)
    a starboard pass corresponds to a positive cross product:

        cross = rel_pos[0] * rel_vel[1] - rel_pos[1] * rel_vel[0]

    cross > 0  =>  relative velocity is clockwise of bearing  =>  starboard pass
    cross < 0  =>  counter-clockwise                          =>  port pass
    cross = 0  =>  head-on along bearing, included as admissible

    The gate: only converging candidates (dot(rel_vel, rel_pos) > 0) are
    constrained.  Diverging candidates are unconditionally admissible —
    no pass-side constraint is needed when the range is opening.

    Parameters
    ----------
    candidates : (N, 2) array of (vn, ve) candidate velocities
    own_pos    : own vessel position [m], shape (2,)
    obs_pos    : obstacle position [m], shape (2,)
    obs_vel    : obstacle velocity [m/s], shape (2,)

    Returns
    -------
    mask : (N,) bool array, True = starboard pass or diverging
    """
    rel_pos    = obs_pos - own_pos
    rel_vel    = candidates - obs_vel[np.newaxis, :]   # (N, 2)
    converging = (rel_vel @ rel_pos) > 0               # (N,)
    cross      = rel_pos[0] * rel_vel[:, 1] - rel_pos[1] * rel_vel[:, 0]
    return ~converging | (cross >= 0.0)
