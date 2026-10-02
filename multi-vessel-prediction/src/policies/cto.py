"""policies/cto_policy.py

Coordinated Turn Obstacle (CTO) policy for maritime collision avoidance.

Based on the action-space obstacle formulation described in:
    Skarø (2026) "Intent-Aware Trajectory Prediction for Multi-Vessel
    Maritime Navigation", Section 5 & 8.

Core idea
---------
Rather than working in Cartesian velocity space (vn, ve) as standard VO
does, this policy operates in the linearised CT action space:

    u = (u1, u2) = (v, v*omega)

where v is speed and omega is turn rate.  Under the linearised CT arc
(second-order Taylor in omega*t), the set of actions that cause a collision
with vessel j at TCPA t* maps exactly to an ellipse in (u1, u2) space:

    E_r(t*) = { u : ((u1-c1)/a1)^2 + ((u2-c2)/a2)^2 <= 1 }

with semi-axes:
    a1 = r_ij / t*          (speed axis)
    a2 = 2*r_ij / (t*)^2   (turn-rate axis)

and center:
    c1 = -(h_hat  . b_ij) / t*
    c2 = -2*(h_hat_perp . b_ij) / (t*)^2

where b_ij = (p_own - p_obs) - A_j(t*) u_j is the relative position at
TCPA under the obstacle's motion, and h_hat / h_hat_perp are the heading
unit vector and its perpendicular.

A second concentric ellipse E_D at the same center but scaled by D/r_ij
encodes the desired clearance D (d_pass).

Action selection (Section 8)
-----------------------------
1. TCPA pre-filter: if TCPA > tcpa_threshold, hold course (u_nom).
2. Compute u_nom = (v_current, 0) — nominal action (straight ahead).
3. If u_nom is outside E_D (clearance ellipse) -> hold course.
4. If u_nom is inside E_D -> project onto the boundary dE_D via the
   unit-circle normalisation trick in scaled coordinates:
       u_tilde = D^{-1} (u_nom - c)
       u_pref  = c + D * u_tilde / ||u_tilde||
5. Return CTAction(u1, u2) — consumed by CTAdapter.

The entire procedure is analytical — no grid search, no iteration.
Cost per obstacle is O(1).

Notes
-----
- NED frame throughout.
- Only handles one obstacle at a time in the current version.  With
  multiple obstacles the ellipses must be intersected or the most
  constraining one selected.
- No COLREGs half-plane constraint in this version (H_ij = R^2).
- Returns CTAction(u1, u2) — pair with CTAdapter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.vessel.state import VesselState, CTAction, wrap_angle
from src.vessel.base import BasePolicy
from src.policies.debug_types import CTODebugFrame, EllipseDebugData


# ---------------------------------------------------------------------------
# Forbidden ellipse
# ---------------------------------------------------------------------------

@dataclass
class ForbiddenEllipse:
    """Forbidden collision ellipse in (u1, u2) action space.

    Parameters
    ----------
    c1, c2     : ellipse center in (u1, u2) space
    a1, a2     : semi-axes (speed axis, turn-rate axis)
    """
    c1: float
    c2: float
    a1: float
    a2: float

    def contains(self, u1: float, u2: float) -> bool:
        """Return True if action (u1, u2) is inside this ellipse."""
        return (((u1 - self.c1) / self.a1) ** 2 +
                ((u2 - self.c2) / self.a2) ** 2) <= 1.0

    def project(self, u1: float, u2: float) -> tuple[float, float]:
        """Project (u1, u2) onto the ellipse boundary via unit-circle trick.

        Transforms to scaled coordinates where the ellipse is a unit circle,
        normalises, then transforms back.  The result is the nearest point on
        the boundary in the scaled metric.

        Parameters
        ----------
        u1, u2 : action to project (assumed inside the ellipse)

        Returns
        -------
        (u1_proj, u2_proj) : point on the ellipse boundary
        """
        # Scaled coordinates: ellipse -> unit circle
        u1_tilde = (u1 - self.c1) / self.a1
        u2_tilde = (u2 - self.c2) / self.a2

        norm = np.sqrt(u1_tilde ** 2 + u2_tilde ** 2)

        if norm < 1e-9:
            # u is at the center — project outward along u1 axis
            return self.c1 + self.a1, self.c2

        # Normalise in scaled space, transform back
        u1_proj = self.c1 + self.a1 * (u1_tilde / norm)
        u2_proj = self.c2 + self.a2 * (u2_tilde / norm)

        return u1_proj, u2_proj


def _build_ellipse(
    own_pos:    np.ndarray,   # (2,) NED [m]
    own_psi:    float,        # own heading [rad]
    obs_pos:    np.ndarray,   # (2,) NED [m]
    obs_vel:    np.ndarray,   # (2,) NED [m/s]
    obs_omega:  float,        # obstacle yaw rate [rad/s], positive = CW (NED)
    tcpa:       float,        # time to CPA [s]
    r:          float,        # combined radius / clearance distance [m]
) -> ForbiddenEllipse:
    """Construct the forbidden ellipse for one obstacle at a given TCPA.

    The obstacle's future position is predicted using the linearised CT arc
    (equation 7 in Skarø 2026) with its observed speed and turn rate:

        A_j(t) u_j = t * v_j * h_j  +  (t^2/2) * v_j * omega_j * h_j_perp

    where h_j is the obstacle heading unit vector and h_j_perp is its
    port-side perpendicular.  This reduces to the standard CV/VO assumption
    when obs_omega = 0.

    Parameters
    ----------
    own_pos   : own vessel position [m]
    own_psi   : own vessel heading [rad], NED convention
    obs_pos   : obstacle position [m]
    obs_vel   : obstacle velocity [m/s]
    obs_omega : obstacle yaw rate [rad/s], positive clockwise (NED)
    tcpa      : time to CPA [s]  (must be > 0)
    r         : radius of the forbidden / clearance disc [m]
    """
    t = tcpa

    # Own heading unit vectors
    h      = np.array([np.cos(own_psi), np.sin(own_psi)])    # forward
    h_perp = np.array([-np.sin(own_psi), np.cos(own_psi)])   # port (left)

    # b_ij(t*): relative position at TCPA under obstacle's CT motion.
    # Linearised CT arc for obstacle (equation 7):
    #   A_j(t) u_j = t * obs_vel  +  (t^2/2) * omega_j * obs_vel_perp
    # where obs_vel_perp = obs_vel rotated 90 deg to port (CCW in NED).
    # b_ij = (p_own - p_obs) - A_j(t*) u_j   (equation 9)
    obs_vel_perp = np.array([-obs_vel[1], obs_vel[0]])        # 90 deg CCW
    A_j_u_j      = obs_vel * t + (t ** 2 / 2.0) * obs_omega * obs_vel_perp
    b             = (own_pos - obs_pos) - A_j_u_j             # (2,)

    # Ellipse center (equations 12)
    c1 = -float(np.dot(h,      b)) / t
    c2 = -2.0 * float(np.dot(h_perp, b)) / (t ** 2)

    # Semi-axes (equation 15)
    a1 = r / t
    a2 = 2.0 * r / (t ** 2)

    return ForbiddenEllipse(c1=c1, c2=c2, a1=a1, a2=a2)


# ---------------------------------------------------------------------------
# TCPA computation
# ---------------------------------------------------------------------------

def _compute_tcpa_cv(
    rel_pos: np.ndarray,   # p_obs - p_own  [m]
    rel_vel: np.ndarray,   # v_own - v_obs  [m/s]
) -> float:
    """CV-based TCPA [s]: time to CPA assuming constant velocities.
    Clamped to 0 if vessels are already diverging.
    """
    rel_speed_sq = float(np.dot(rel_vel, rel_vel))
    if rel_speed_sq < 1e-12:
        return 0.0
    return max(float(np.dot(rel_pos, rel_vel) / rel_speed_sq), 0.0)


def _compute_tcpa_ct(
    own_pos:   np.ndarray,
    own_psi:   float,
    own_speed: float,
    own_omega: float,
    obs_pos:   np.ndarray,
    obs_vel:   np.ndarray,
    obs_omega: float,
    t_max:     float = 600.0,
    n_init:    int   = 50,
) -> float:
    """CT-based TCPA [s]: time of closest approach under linearised CT arcs.

    Minimises ||r(t)||^2 where:

        r(t) = d0
             + t*v_own*h_own + (t^2/2)*v_own*omega_own*h_own_perp
             - t*obs_vel     - (t^2/2)*obs_omega*obs_vel_perp

    ||r(t)||^2 is degree-4 in t so d/dt is degree-3.  We find the minimum
    by evaluating on a coarse grid then refining with a bounded scalar
    minimiser.

    Parameters
    ----------
    own_pos   : own vessel position [m]
    own_psi   : own vessel heading [rad]
    own_speed : own vessel speed [m/s]
    own_omega : own yaw rate [rad/s] (use 0 before action is selected)
    obs_pos   : obstacle position [m]
    obs_vel   : obstacle velocity [m/s]
    obs_omega : obstacle yaw rate [rad/s]
    t_max     : search horizon [s]
    n_init    : coarse grid resolution

    Returns
    -------
    tcpa : time of closest approach [s], clamped to 0 if diverging
    """
    from scipy.optimize import minimize_scalar

    d0 = own_pos - obs_pos

    h_own      = np.array([np.cos(own_psi), np.sin(own_psi)])
    h_own_perp = np.array([-np.sin(own_psi), np.cos(own_psi)])
    obs_vel_perp = np.array([-obs_vel[1], obs_vel[0]])

    def dist_sq(t: float) -> float:
        own_arc = t * own_speed * h_own + (t**2 / 2.0) * own_speed * own_omega * h_own_perp
        obs_arc = t * obs_vel           + (t**2 / 2.0) * obs_omega * obs_vel_perp
        r = d0 + own_arc - obs_arc
        return float(np.dot(r, r))

    # Coarse grid to find the basin of the global minimum
    t_grid = np.linspace(0.0, t_max, n_init)
    d_grid = np.array([dist_sq(t) for t in t_grid])
    best_i = int(np.argmin(d_grid))

    t_lo = t_grid[max(best_i - 1, 0)]
    t_hi = t_grid[min(best_i + 1, n_init - 1)]

    result = minimize_scalar(dist_sq, bounds=(t_lo, t_hi), method="bounded")
    tcpa   = max(float(result.x), 0.0)

    # Already diverging: distance increasing immediately from t=0
    if best_i == 0 and dist_sq(1e-3) > dist_sq(0.0):
        return 0.0

    return tcpa


# ---------------------------------------------------------------------------
# CTOPolicy
# ---------------------------------------------------------------------------

class CTOPolicy(BasePolicy):
    """Coordinated Turn Obstacle policy without COLREGs constraints.

    Returns CTAction(u1, u2) — pair with CTAdapter.

    Parameters
    ----------
    tcpa_threshold : float
        On/off switch [s].  If all obstacles have TCPA >= tcpa_threshold
        the policy is inactive and holds nominal course.
        Maps to VesselInit.tcpa_threshold.
    d_pass : float
        Desired passing clearance [m].  Defines the clearance ellipse E_D.
        Maps to VesselInit.d_pass.
    """

    def __init__(
        self,
        nominal_psi:    float,
        ref_speed:      float,
        tcpa_threshold: float = 120.0,
        d_pass:         float = 100.0,
        los_lookahead:  float = 200.0,
        k_psi:          float = 0.5,
        omega_max:      float = np.pi / 4,
        tcpa_method:    str   = "ct",    # "cv" or "ct"
    ) -> None:
        if tcpa_method not in ("cv", "ct"):
            raise ValueError(f"tcpa_method must be 'cv' or 'ct', got {tcpa_method!r}")
        self.ref_speed      = ref_speed
        self.tcpa_threshold = tcpa_threshold
        self.d_pass         = d_pass
        self.los_lookahead  = los_lookahead
        self.k_psi          = k_psi
        self.omega_max      = omega_max
        self.tcpa_method    = tcpa_method

        self._v_pref_unit  = np.array([np.cos(nominal_psi), np.sin(nominal_psi)])
        self._origin_pos: np.ndarray | None = None
        self.last_debug: CTODebugFrame | None = None

    # ------------------------------------------------------------------
    # BasePolicy interface
    # ------------------------------------------------------------------

    def select_action(
        self,
        own_state:    VesselState,
        other_states: list[VesselState],
    ) -> CTAction:
        """Select an action using the CTO analytic projection.

        Steps
        -----
        1. Compute TCPA for each obstacle.  Filter to threats
           (TCPA < tcpa_threshold).  If none -> hold course.
        2. u_nom = (v_current, 0) — straight ahead at current speed.
        3. For each threat build the clearance ellipse E_D.
        4. Find the most constraining threat (smallest TCPA among those
           whose E_D contains u_nom).
        5. If u_nom is outside all E_D ellipses -> hold course.
        6. Project u_nom onto dE_D of the most constraining ellipse.
        7. Return CTAction(u1, u2).
        """
        own_pos = own_state.position
        own_vel = np.array([own_state.vn, own_state.ve])
        own_psi = own_state.psi
        v_cur   = own_state.speed

        # Latch origin on first call
        if self._origin_pos is None:
            self._origin_pos = own_pos.copy()

        # LOS guidance: cross-track error -> desired heading -> nominal turn rate
        perp      = np.array([-self._v_pref_unit[1], self._v_pref_unit[0]])
        e         = -np.dot(own_pos - self._origin_pos, perp)
        psi_d     = (np.arctan2(self._v_pref_unit[1], self._v_pref_unit[0])
                     + np.arctan2(e, self.los_lookahead))
        omega_nom = float(np.clip(self.k_psi * wrap_angle(psi_d - own_psi),
                                  -self.omega_max, self.omega_max))

        # Nominal action: reference speed + LOS turn rate
        u_nom = (self.ref_speed, self.ref_speed * omega_nom)

        # --- 1. TCPA pre-filter -------------------------------------------
        threats: list[tuple[float, VesselState]] = []   # (tcpa, obs)
        for obs in other_states:
            obs_vel = np.array([obs.vn, obs.ve])
            if self.tcpa_method == "ct":
                tcpa = _compute_tcpa_ct(
                    own_pos   = own_pos,
                    own_psi   = own_psi,
                    own_speed = v_cur,
                    own_omega = own_state.r,
                    obs_pos   = obs.position,
                    obs_vel   = obs_vel,
                    obs_omega = obs.r,
                )
            else:
                tcpa = _compute_tcpa_cv(
                    rel_pos = obs.position - own_pos,
                    rel_vel = own_vel - obs_vel,
                )
            if 0.0 < tcpa < self.tcpa_threshold:
                threats.append((tcpa, obs))

        if not threats:
            self.last_debug = CTODebugFrame(
                u_nom=u_nom, u_projected=u_nom, u_selected=u_nom,
                ref_speed=self.ref_speed, omega_max=self.omega_max,
            )
            return CTAction(speed=self.ref_speed, turn_rate=omega_nom)

        # --- 2 & 3. build clearance ellipse E_D per threat ----------------
        # Find the most constraining threat: smallest TCPA whose E_D
        # contains u_nom.  This is the threat that requires action first.
        all_ellipses: list[EllipseDebugData] = []
        constraining: list[tuple[float, ForbiddenEllipse]] = []

        for tcpa, obs in sorted(threats, key=lambda x: x[0]):
            obs_vel = np.array([obs.vn, obs.ve])
            ellipse_D = _build_ellipse(
                own_pos   = own_pos,
                own_psi   = own_psi,
                obs_pos   = obs.position,
                obs_vel   = obs_vel,
                obs_omega = obs.r,
                tcpa      = tcpa,
                r         = self.d_pass,
            )
            all_ellipses.append(EllipseDebugData(
                c1=ellipse_D.c1, c2=ellipse_D.c2,
                a1=ellipse_D.a1, a2=ellipse_D.a2,
            ))
            if ellipse_D.contains(*u_nom):
                constraining.append((tcpa, ellipse_D))

        # --- 5. u_nom safe w.r.t. all clearance ellipses -> execute nominal ---
        if not constraining:
            self.last_debug = CTODebugFrame(
                u_nom=u_nom, u_projected=u_nom, u_selected=u_nom,
                ellipses=all_ellipses, ref_speed=self.ref_speed, omega_max=self.omega_max,
            )
            return CTAction(speed=self.ref_speed, turn_rate=omega_nom)

        # --- 6. project onto most constraining ellipse (smallest TCPA) ----
        _, ellipse = constraining[0]
        u1_proj, u2_proj = ellipse.project(*u_nom)

        # The scaled-space projection can push u1 well above v_cur (speed-up).
        # When that happens, fix speed at v_cur and solve for the minimum turn
        # rate that places the action on the ellipse boundary instead:
        #   ((v_cur - c1)/a1)^2 + ((u2 - c2)/a2)^2 = 1
        #   => u2 = c2 +/- a2 * sqrt(1 - xi^2)
        if u1_proj > self.ref_speed:
            xi       = (self.ref_speed - ellipse.c1) / ellipse.a1
            delta_u2 = ellipse.a2 * np.sqrt(max(0.0, 1.0 - xi ** 2))
            u2_a     = ellipse.c2 + delta_u2
            u2_b     = ellipse.c2 - delta_u2
            u2_proj  = u2_a if abs(u2_a) <= abs(u2_b) else u2_b
            u1_proj  = self.ref_speed

        # u_projected: ideal boundary point (may have u1 < 0 or |omega| > omega_max)
        u_projected = (u1_proj, u2_proj)

        u1_proj   = max(u1_proj, 0.0)
        turn_rate = u2_proj / u1_proj if u1_proj > 1e-9 else 0.0
        turn_rate = float(np.clip(turn_rate, -self.omega_max, self.omega_max))

        self.last_debug = CTODebugFrame(
            u_nom=u_nom,
            u_projected=u_projected,
            u_selected=(u1_proj, u1_proj * turn_rate),
            ellipses=all_ellipses,
            ref_speed=self.ref_speed,
            omega_max=self.omega_max,
        )
        return CTAction(speed=u1_proj, turn_rate=turn_rate)