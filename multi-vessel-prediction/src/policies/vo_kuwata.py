"""policies/vo_kuwata.py

Grid-search Velocity Obstacle policy for maritime collision avoidance with COLREGs support.

Based on:
    Kuwata et al. (2011) "Safe Maritime Navigation with COLREGS Using
    Velocity Obstacles", IEEE/RSJ IROS.

Core idea
---------
For each obstacle B, a Velocity Obstacle (VO) cone is constructed in the
velocity space of own vessel A.  A candidate velocity v_A is inside the VO
if the relative velocity (v_A - v_B) points into the Minkowski sum disc
(B + -A), i.e. the ray from the origin in direction (v_A - v_B) intersects
the disc of radius d_pass centred at (p_B - p_A).

Algorithm flow
--------------
1. Pre-filter obstacles by TCPA.  Only obstacles with TCPA < tcpa_threshold
   are considered threats.  If no threats exist, return v_pref directly
   (constant course, no VO computation).
2. For each threatening obstacle, determine cone radius and candidate mask
   from the COLREGs role assigned to that obstacle pair.
3. Build one VO cone per threatening obstacle.  For HEAD_ON obstacles the
   primary cone uses RVO geometry (apex shifted to the velocity midpoint),
   encoding the symmetric Rule 14 obligation of both vessels to turn
   starboard.  Fallback cones always use plain VO (apex at v_obs) because
   in the fallback regime cooperative behaviour cannot be assumed.
4. Primary admissibility: candidates inside any cone are inadmissible.
   Score admissible candidates by squared distance to v_pref_now.
   Select the lowest-cost admissible candidate.
5. Fallback admissibility (GIVE_WAY, HEAD_ON, OVERTAKING, AVOID_COLLISION):
   rebuild cones with d_coll.  Project v_pref_now onto the boundary of the
   d_pass cones to obtain v_pref_proj.  Score d_coll-admissible candidates
   using v_pref_proj as the deviation anchor.
   VODebugFrame.fallback_active is set to True in this regime.
   STAND_ON skips fallback and goes directly to full stop.
6. True deadlock: if ALL candidates also violate d_coll, return full stop.

COLREGs role behaviour
----------------------
GIVE_WAY        : primary d_pass, fallback d_coll, starboard candidates only
STAND_ON        : primary d_coll, no fallback (full stop on failure)
HEAD_ON         : primary d_pass (RVO apex), fallback d_coll (VO apex),
                  starboard candidates only
OVERTAKING      : primary d_pass, fallback d_coll, no side constraint
AVOID_COLLISION : primary d_pass, fallback d_coll, no side constraint

When no ColregsGraph is supplied all obstacles default to AVOID_COLLISION.

Starboard constraint is implemented as a hard per-obstacle candidate mask.
For each obstacle requiring a starboard pass, candidates whose relative
velocity crosses to the left of the bearing vector to that obstacle are
removed.  The mask is computed per obstacle at runtime, not globally at
init, since the correct passing side depends on the bearing to each
specific target.

Parameters
----------
tcpa_threshold : float
    Acts as an on/off switch.  If ALL obstacles have TCPA >= tcpa_threshold
    the VO algorithm is inactive and the vessel holds its nominal course.
    Maps to VesselInit.tcpa_threshold.
d_pass : float
    Desired passing distance [m].  Sets the primary cone size.
    Maps to VesselInit.d_pass.
d_coll : float
    Hard collision radius [m].  Must be strictly less than d_pass.
    Used only when d_pass cones cover all candidates (fallback regime).
head_on_rvo_weight : float
    RVO apex interpolation weight applied to HEAD_ON primary cones.
    0.0 = plain VO (apex at v_obs).
    0.5 = full RVO (apex at midpoint of v_own and v_obs), encoding equal
          shared responsibility under COLREGs Rule 14.
    Values between 0 and 0.5 give partial credit for anticipated obstacle
    cooperation — useful when the obstacle may be slow to manoeuvre.
    Only applied during the primary admissibility pass; fallback cones
    always use plain VO (rvo_weight=0.0) regardless of this parameter.

Other notes
-----------
- NED frame throughout.
- Nominal heading and v_pref fixed at construction, never updated.
- Candidate grid built once at init.
"""

from __future__ import annotations

import numpy as np

from src.vessel.state import VesselState, CVAction
from src.vessel.base import BasePolicy
from src.colregs_graph.colregs_graph import VesselRole, ColregsGraph
from src.policies.debug_types import VODebugFrame, VOConeDebugData, StarboardConstraintDebugData

from .vo_common import (
    tcpa,
    VOCone,
    build_vo_cone,
    project_through_cones,
    RoleConfig,
    RolePassDistances,
    role_config,
    starboard_mask,
)


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

def _score_candidates(
    candidates: np.ndarray,
    admissible: np.ndarray,
    v_pref:     np.ndarray,
) -> np.ndarray:
    """Score candidates by squared Euclidean distance to v_pref.

    Inadmissible candidates receive cost = inf.

    Parameters
    ----------
    candidates : (N, 2) velocity grid
    admissible : (N,) bool mask
    v_pref     : anchor velocity — v_pref_now in primary pass,
                 v_pref_proj in fallback pass

    Returns
    -------
    cost : (N,) float array, inf for inadmissible candidates
    """
    cost = np.sum((candidates - v_pref[np.newaxis, :])**2, axis=1)
    return np.where(admissible, cost, np.inf)


# ---------------------------------------------------------------------------
# VOPolicyKuwata
# ---------------------------------------------------------------------------

class VOPolicyKuwata(BasePolicy):
    """Velocity Obstacle policy with COLREGs role-based constraints.

    Returns CVAction(vn, ve) — pair with CVAdapter.

    Each obstacle is passed as a (VesselState, VesselRole) pair.  The role
    determines the cone radius, fallback behaviour, and side constraint
    applied for that specific vessel pair.

    When colregs_graph is None all obstacles default to AVOID_COLLISION,
    giving plain VO behaviour with no rule-based constraints.

    Build it with prediction.vo_rollout.make_vo_policy(VOAgentConfig), which
    holds the parameter defaults.

    Parameters
    ----------
    nominal_psi : float
        Fixed nominal heading [rad], 0 = north, clockwise positive (NED).
    ref_speed : float
        Reference speed [m/s].
    tcpa_threshold : float
        On/off switch [s].  Obstacles with TCPA >= tcpa_threshold are ignored.
        Maps to VesselInit.tcpa_threshold.
    d_pass : float
        Desired passing distance [m].  Sets the primary cone size.
        Maps to VesselInit.d_pass.
    d_coll : float
        Hard collision radius [m].  Must be < d_pass.  Used only in fallback.
    max_heading_change : float
        Half-width of heading search range [rad] around nominal_psi.
    n_speeds : int
        Speed levels in [0, ref_speed].
    n_headings : int
        Heading samples across +/-max_heading_change.
    los_lookahead : float
        LOS guidance lookahead distance [m].
    head_on_rvo_weight : float
        RVO apex weight for HEAD_ON primary cones.  0.5 = full RVO (equal
        shared responsibility).  Applied only during the primary pass;
        fallback cones always use plain VO.
    colregs_graph : ColregsGraph | None
        Pairwise COLREGs role graph.  None → all roles default to
        AVOID_COLLISION (plain VO, no rule-based constraints).
    own_vessel_idx : int
        Index of this vessel in the graph (0 = ego).
    other_vessel_indices : list[int] | None
        Indices of the other vessels in the same order as other_states
        passed to select_action.  None → sequential 0, 1, 2, ...
    """

    def __init__(
        self,
        *,
        nominal_psi:          float,
        ref_speed:            float,
        tcpa_threshold:       float,
        d_pass:               float,
        d_coll:               float,
        max_heading_change:   float,
        n_speeds:             int,
        n_headings:           int,
        los_lookahead:        float,
        head_on_rvo_weight:   float,
        origin_pos:           np.ndarray | None        = None,
        d_pass_roles:         RolePassDistances | None = None,
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

        self.nominal_psi          = nominal_psi
        self.ref_speed            = ref_speed
        self.tcpa_threshold       = tcpa_threshold
        self.d_pass               = d_pass
        self.d_coll               = d_coll
        self._d_pass_r            = (
            d_pass_roles if d_pass_roles is not None
            else RolePassDistances.from_scalars(d_pass, d_coll)
        )
        self.max_heading_change   = max_heading_change
        self.n_speeds             = n_speeds
        self.n_headings           = n_headings
        self.los_lookahead        = los_lookahead
        self.head_on_rvo_weight   = head_on_rvo_weight

        self._v_pref      = np.array([
            ref_speed * np.cos(nominal_psi),
            ref_speed * np.sin(nominal_psi),
        ])
        self._v_pref_unit = self._v_pref / np.linalg.norm(self._v_pref)
        self._origin_pos  = origin_pos.copy() if origin_pos is not None else None
        self._candidates  = self._build_candidates()
        self.last_debug: VODebugFrame | None = None

        self._colregs_graph        = colregs_graph
        self._own_vessel_idx       = own_vessel_idx
        self._other_vessel_indices = other_vessel_indices

    def set_colregs_graph(self, graph: ColregsGraph | None) -> None:
        """Replace the COLREGs graph (e.g. from a live classifier)."""
        self._colregs_graph = graph

    # ------------------------------------------------------------------
    # COLREGs role annotation
    # ------------------------------------------------------------------

    def _annotate_roles(
        self,
        other_states: list[VesselState],
    ) -> list[tuple[VesselState, VesselRole, bool]]:
        """Attach a VesselRole and graph-edge flag to each other vessel.

        The bool is True when the vessel has an edge in the COLREGs graph,
        which is used to bypass the TCPA pre-filter and keep cones persistent.
        """
        if self._colregs_graph is None:
            return [(s, VesselRole.AVOID_COLLISION, False) for s in other_states]

        role_map = dict(self._colregs_graph.get_roles(self._own_vessel_idx))
        indices  = (
            self._other_vessel_indices
            if self._other_vessel_indices is not None
            else list(range(len(other_states)))
        )
        return [
            (obs, role_map.get(idx, VesselRole.AVOID_COLLISION), idx in role_map)
            for obs, idx in zip(other_states, indices)
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

            # --- LOS guidance: steer back toward track line -------------------
            e         = -np.dot(own_pos - self._origin_pos,
                                np.array([-self._v_pref_unit[1], self._v_pref_unit[0]]))
            psi_d     = (np.arctan2(self._v_pref_unit[1], self._v_pref_unit[0])
                        + np.arctan2(e, self.los_lookahead))
            v_pref_now = self.ref_speed * np.array([np.cos(psi_d), np.sin(psi_d)])

            # --- 1. pre-filter: keep only threatening obstacles ---------------
            # Vessels with a COLREGs graph edge are always included so that
            # their VO cones remain persistent (the graph has hysteresis).
            threats: list[tuple[VesselState, VesselRole]] = []
            for obs, role, in_graph in annotated:
                rel_pos = obs.position - own_pos
                obs_vel = np.array([obs.vn, obs.ve])
                t       = tcpa(rel_pos, own_vel - obs_vel)
                if in_graph or t < self.tcpa_threshold:
                    threats.append((obs, role))

            if not threats:
                self.last_debug = VODebugFrame(
                    v_pref_vn=float(v_pref_now[0]), v_pref_ve=float(v_pref_now[1]),
                    v_sel_vn=float(v_pref_now[0]),  v_sel_ve=float(v_pref_now[1]),
                    cones=[], ref_speed=self.ref_speed,
                    fallback_active=False,
                )
                return CVAction(vn=float(v_pref_now[0]), ve=float(v_pref_now[1]))

            # --- 2. build per-obstacle configs and primary cones --------------
            configs: list[RoleConfig] = [
                role_config(role, self._d_pass_r, self.d_coll)
                for _, role in threats
            ]

            # Primary cones: HEAD_ON obstacles use RVO geometry (shifted apex)
            # to encode the symmetric Rule 14 obligation.  All other roles use
            # plain VO (rvo_weight=0.0).
            cones_primary: list[VOCone] = [
                build_vo_cone(
                    own_pos    = own_pos,
                    obs_pos    = obs.position,
                    obs_vel    = np.array([obs.vn, obs.ve]),
                    combined_r = cfg.primary_r,
                    own_vel    = own_vel if role is VesselRole.HEAD_ON else None,
                    rvo_weight = self.head_on_rvo_weight if role is VesselRole.HEAD_ON else 0.0,
                )
                for (obs, role), cfg in zip(threats, configs)
            ]

            debug_cones = [VOConeDebugData(
                apex_vn=c.apex[0], apex_ve=c.apex[1],
                axis_n=c.axis[0],  axis_e=c.axis[1],
                half_angle=c.half_angle,
            ) for c in cones_primary]

            # --- 3. primary admissibility -------------------------------------
            # The debug_starboard list and inadmissible mask are built together
            # so the laterally-shifted sb_ref_vel is computed only once per
            # obstacle and stored consistently in both.
            candidates   = self._candidates
            inadmissible = np.zeros(len(candidates), dtype=bool)
            debug_starboard: list[StarboardConstraintDebugData] = []

            for (obs, role), cone, cfg in zip(threats, cones_primary, configs):
                in_cone = cone.contains_batch(candidates)
                inadmissible |= in_cone

                if cfg.starboard_only:
                    obs_vel_arr = np.array([obs.vn, obs.ve])
                    bearing     = obs.position - own_pos
                    bearing     = bearing / np.linalg.norm(bearing)
                    lateral     = np.array([-bearing[1], bearing[0]])

                    # For HEAD_ON, shift the starboard constraint reference
                    # laterally by the anticipated cooperative manoeuvre of the
                    # obstacle (rvo_weight * lateral component of v_own - v_obs).
                    # This makes the constraint more conservative — it accounts
                    # for the fact that the RVO cone shift may have opened up
                    # candidates that would produce a close pass if the obstacle
                    # does not cooperate.  The axial component is excluded because
                    # it does not affect passing side geometry.
                    if role is VesselRole.HEAD_ON:
                        rvo_shift     = self.head_on_rvo_weight * (own_vel - obs_vel_arr)
                        lateral_shift = np.dot(rvo_shift, lateral) * lateral
                        sb_ref_vel    = obs_vel_arr + lateral_shift
                    else:
                        sb_ref_vel = obs_vel_arr

                    sb_mask = starboard_mask(
                        candidates = candidates,
                        own_pos    = own_pos,
                        obs_pos    = obs.position,
                        obs_vel    = sb_ref_vel,
                    )
                    inadmissible |= ~sb_mask

                    debug_starboard.append(StarboardConstraintDebugData(
                        apex_vn=float(sb_ref_vel[0]), apex_ve=float(sb_ref_vel[1]),
                        bearing_n=float(bearing[0]),  bearing_e=float(bearing[1]),
                    ))

            # Forward half-plane: exclude candidates going backward.
            # dot(v, v_pref_unit) >= 0 — zero-speed candidates pass (dot = 0).
            forward_mask       = (candidates @ self._v_pref_unit) >= 0.0
            admissible_primary = ~inadmissible & forward_mask

            if admissible_primary.any():
                cost = _score_candidates(
                    candidates = candidates,
                    admissible = admissible_primary,
                    v_pref     = v_pref_now,
                )
                best = candidates[int(np.argmin(cost))]
                self.last_debug = VODebugFrame(
                    v_pref_vn=float(v_pref_now[0]), v_pref_ve=float(v_pref_now[1]),
                    v_sel_vn=float(best[0]),        v_sel_ve=float(best[1]),
                    cones=debug_cones, ref_speed=self.ref_speed,
                    fallback_active=False,
                    starboard_constraints=debug_starboard,
                    forward_unit_n=float(self._v_pref_unit[0]),
                    forward_unit_e=float(self._v_pref_unit[1]),
                )
                return CVAction(vn=float(best[0]), ve=float(best[1]))

            # --- 4. Partition threats ----------------------------------------
            # soft_threats: all but STAND_ON — relax primary d_pass → d_coll.
            # STAND_ON threats keep their primary cones (cones_hard_coll below).
            soft_threats = [
                (obs, cfg) for (obs, _), cfg in zip(threats, configs)
                if cfg.has_fallback
            ]

            if not soft_threats:
                self.last_debug = VODebugFrame(
                    v_pref_vn=float(v_pref_now[0]), v_pref_ve=float(v_pref_now[1]),
                    v_sel_vn=0.0, v_sel_ve=0.0,
                    cones=debug_cones, ref_speed=self.ref_speed,
                    fallback_active=True,
                    starboard_constraints=debug_starboard,
                )
                return CVAction(vn=0.0, ve=0.0)

            # --- 5. Build fallback cones ------------------------------------
            # Soft threats: relax to d_coll (fresh cones, plain VO)
            cones_soft_coll: list[VOCone] = [
                build_vo_cone(
                    own_pos    = own_pos,
                    obs_pos    = obs.position,
                    obs_vel    = np.array([obs.vn, obs.ve]),
                    combined_r = self.d_coll,
                )
                for obs, _ in soft_threats
            ]

            # Hard threats: reuse primary cones (already d_coll, no relaxation)
            cones_hard_coll = [
                cone for (obs, _), cfg, cone in zip(threats, configs, cones_primary)
                if not cfg.has_fallback
            ]

            # --- 6. Compute fallback anchor ---------------------------------
            # Reuse primary cones for soft threats as d_pass projection set
            cones_pass_soft = [
                cone for (obs, _), cfg, cone in zip(threats, configs, cones_primary)
                if cfg.has_fallback
            ]
            v_pref_proj = project_through_cones(v_pref_now, cones_pass_soft)

            # Push v_pref_proj onto the starboard half-plane if it violates it
            for obs, cfg in soft_threats:
                if not cfg.starboard_only:
                    continue
                rel_pos     = obs.position - own_pos
                obs_vel_arr = np.array([obs.vn, obs.ve])
                rel_vel     = v_pref_proj - obs_vel_arr
                if np.dot(rel_vel, rel_pos) <= 0.0:
                    continue
                cross = rel_pos[0] * rel_vel[1] - rel_pos[1] * rel_vel[0]
                if cross < 0.0:
                    c_perp      = np.array([-rel_pos[1], rel_pos[0]], dtype=float)
                    c_normal    = c_perp / np.linalg.norm(c_perp)
                    offset      = float(np.dot(c_normal, obs_vel_arr))
                    dist        = float(np.dot(c_normal, v_pref_proj)) - offset
                    v_pref_proj = v_pref_proj - dist * c_normal

            # Guard: if anchor is still inside a fallback cone, reset to v_pref_now
            all_fallback_cones = cones_soft_coll + cones_hard_coll
            if any(c.contains(v_pref_proj) for c in all_fallback_cones):
                v_pref_proj = v_pref_now.copy()

            # --- 7. Fallback admissibility ----------------------------------
            inadmissible_coll = np.zeros(len(candidates), dtype=bool)

            # Hard threat cones (STAND_ON d_coll — no starboard constraint for STAND_ON)
            for cone in cones_hard_coll:
                in_cone = cone.contains_batch(candidates)
                inadmissible_coll |= in_cone

            # Soft threat cones with starboard constraint (raw obs_vel, no RVO shift)
            for (obs, cfg), cone in zip(soft_threats, cones_soft_coll):
                in_cone = cone.contains_batch(candidates)
                inadmissible_coll |= in_cone
                if cfg.starboard_only:
                    sb_mask = starboard_mask(
                        candidates = candidates,
                        own_pos    = own_pos,
                        obs_pos    = obs.position,
                        obs_vel    = np.array([obs.vn, obs.ve]),
                    )
                    inadmissible_coll |= ~sb_mask

            # Forward half-plane preserved from primary
            admissible_coll = ~inadmissible_coll & forward_mask

            if admissible_coll.any():
                cost = _score_candidates(
                    candidates = candidates,
                    admissible = admissible_coll,
                    v_pref     = v_pref_proj,
                )
                best = candidates[int(np.argmin(cost))]
                self.last_debug = VODebugFrame(
                    v_pref_vn=float(v_pref_now[0]),  v_pref_ve=float(v_pref_now[1]),
                    v_sel_vn=float(best[0]),         v_sel_ve=float(best[1]),
                    cones=debug_cones, ref_speed=self.ref_speed,
                    fallback_active=True,
                    v_pref_proj_vn=float(v_pref_proj[0]),
                    v_pref_proj_ve=float(v_pref_proj[1]),
                    starboard_constraints=debug_starboard,
                )
                return CVAction(vn=float(best[0]), ve=float(best[1]))

            # --- 8. True deadlock: full stop --------------------------------
            self.last_debug = VODebugFrame(
                v_pref_vn=float(v_pref_now[0]),  v_pref_ve=float(v_pref_now[1]),
                v_sel_vn=0.0, v_sel_ve=0.0,
                cones=debug_cones, ref_speed=self.ref_speed,
                fallback_active=True,
                v_pref_proj_vn=float(v_pref_proj[0]),
                v_pref_proj_ve=float(v_pref_proj[1]),
                starboard_constraints=debug_starboard,
            )
            return CVAction(vn=0.0, ve=0.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_candidates(self) -> np.ndarray:
        """Build fixed (n_speeds * n_headings, 2) candidate array."""
        headings = self.nominal_psi + np.linspace(
            -self.max_heading_change, self.max_heading_change, self.n_headings
        )
        speeds = np.linspace(0.0, self.ref_speed, self.n_speeds)
        rows = []
        for spd in speeds:
            for hdg in headings:
                rows.append([spd * np.cos(hdg), spd * np.sin(hdg)])
        return np.array(rows, dtype=float)