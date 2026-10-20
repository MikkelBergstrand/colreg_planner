"""prediction/vo_rollout.py

VO rollout for multi-vessel trajectory prediction.

Each vessel is assumed to act as a VOPolicyAnalytic (or VOPolicyKuwata) agent.
At every rollout step all vessels select their action simultaneously from the
same state snapshot, then all states are propagated forward with the motion
model named by rollout_model.  The
result is a set of predicted trajectories — one per vessel — spanning
n_steps * dt seconds.

Typical usage
-------------
    from colregs_planner.prediction.vo_rollout import vo_rollout, VOAgentConfig

    trajectories = vo_rollout(
        initial_states = {vid: vessel.state for vid, vessel in ...},
        dt      = 1.0,
        n_steps = 60,
    )
    # trajectories[vid] is a list of n_steps+1 VesselState objects

Per-vessel VO parameters
------------------------
Pass a dict of VOAgentConfig objects to customise each vessel's policy.
Any vessel without an entry uses the default config.  nominal_psi and
ref_speed are inferred from the vessel's initial state when not given.

policy_type selects the solver: "vo_analytic" (VOPolicyAnalytic, the
paper's solver) or "vo_kuwata" (VOPolicyKuwata, the grid-search VO of
Kuwata et al. 2011).  The analytic policy uses course_budget_deg,
achievable_ratio and continuity_weight; the Kuwata policy uses
max_heading_change, n_speeds and n_headings.

Notes
-----
- Fresh policy and motion-model instances are created for each vessel at the
  start of the rollout so the caller's objects are never mutated.
- ref_speed must be > 0 for all vessels (both policies require a non-zero
  preferred velocity for LOS guidance).
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, fields, replace
from itertools import product as itertools_product
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np

from colregs_planner.vessel.state import Intent, VesselState, InitialState, CVAction
from colregs_planner.vessel.base import BaseSimModel
from colregs_planner.policies.vo_kuwata import VOPolicyKuwata
from colregs_planner.policies.vo_analytic import VOPolicyAnalytic
from colregs_planner.policies.vo_common import RolePassDistances
from colregs_planner.colregs_graph.colregs_graph import ColregsGraph, apply_responsiveness
from colregs_planner.colregs_graph.colregs_classifier import ColregsClassifier
from colregs_planner.motion_models.adapters import build_adapter

if TYPE_CHECKING:
    from colregs_planner.prediction.parallel import Pending, RolloutPool
    from colregs_planner.seachart.hazard import HazardMap


# ---------------------------------------------------------------------------
# Per-vessel configuration
# ---------------------------------------------------------------------------

@dataclass
class VOAgentConfig:
    """VO policy parameters for one vessel in the rollout.

    The field defaults below are the only place VO parameter defaults are
    set: YAML blocks go through from_dict(), and every VO policy is built
    from a VOAgentConfig by make_vo_policy().

    nominal_psi and ref_speed are inferred from the vessel's initial state
    when left as None.  d_coll (the minimum passing distance) defaults to
    d_coll_ratio * d_pass when left as None; use resolved_d_coll().

    origin_north / origin_east fix the LOS track-line origin to the vessel's
    mission start position so the rollout predicts a return to the original
    track, not a fresh track from wherever the vessel currently is.  When
    both are None the origin latches on the first rollout step as usual.

    policy_type selects the underlying VO implementation:
      "vo_analytic" -- analytic VOPolicyAnalytic (default)
      "vo_kuwata"   -- grid-search VOPolicyKuwata

    Kuwata-only params (max_heading_change, n_speeds, n_headings) are ignored
    when policy_type="vo_analytic"; course_budget_deg, achievable_ratio and
    continuity_weight are analytic-only.  head_on_rvo_weight applies to both.
    """
    policy_type:        str             = "vo_analytic"
    tcpa_threshold:     float           = 120.0
    d_pass:             float                    = 100.0
    d_coll:             Optional[float]          = None              # None -> d_coll_ratio * d_pass (see resolved_d_coll)
    d_coll_ratio:       float                    = 0.1               # default minimum passing distance as a fraction of d_pass
    d_pass_roles:       RolePassDistances | None = None              # per-role override; None → from_scalars(d_pass, d_coll)
    max_heading_change: float           = np.pi / 8 
    n_speeds:           int             = 8
    n_headings:         int             = 72
    head_on_rvo_weight: float           = 0.5
    rollout_model:      str             = "ct_decoupled_ar"
    k_psi:              float           = 0.1
    tau_speed:          float           = 0.0     # rollout speed response time constant [s]; 0 = instant
    los_lookahead:      float           = 100.0
    course_budget_deg:  float           = 70.0      # vo_analytic: largest course alteration from nominal [deg]; ~60 recommended, 180 = no limit
    achievable_ratio:   float           = 1       # vo_analytic: fraction alpha of the best reachable passing distance demanded when d_pass is out of reach; keep slightly below 1 (e.g. 0.99)
    continuity_weight:  float           = 0.9    # vo_analytic: lambda in Eq. 19, pull toward the previously selected heading (weight 1 = preferred heading)
    terrain_lookahead_s: float          = 30.0   # vo_analytic with hazards: a heading is blocked if it comes within
    terrain_clearance:  float           = 5.0    #   terrain_clearance [m] of a hazard within this many seconds
    nominal_psi:        Optional[float]           = None   # inferred from initial state
    ref_speed:          Optional[float]           = None   # inferred from initial state
    origin_north:       Optional[float]           = None   # mission start north [m]
    origin_east:        Optional[float]           = None   # mission start east  [m]
    pref_vel_seq:       Optional[np.ndarray]      = None   # (n_steps, 2) nominal [vn,ve] per rollout step
    hazards:            Optional["HazardMap"]     = None   # land and shoals (vo_analytic only)

    def resolved_d_coll(self) -> float:
        """Explicit d_coll if set, else d_coll_ratio * d_pass."""
        return self.d_coll if self.d_coll is not None else self.d_pass * self.d_coll_ratio

    @classmethod
    def from_dict(
        cls,
        d:        dict,
        ignore:   Iterable[str] = (),
        **overrides,
    ) -> "VOAgentConfig":
        """Build from a YAML block.

        Only keys present in d override the field defaults.  Keys that are
        neither a tunable field nor listed in ignore raise a ValueError, so
        typos and removed parameters do not pass silently.  overrides (e.g.
        the runtime fields nominal_psi, ref_speed, origin_*) take precedence
        over d.  d_pass_roles may be given as a dict of per-role distances.
        """
        tunable = {f.name for f in fields(cls)} - _RUNTIME_FIELDS
        unknown = set(d) - tunable - set(ignore)
        if unknown:
            raise ValueError(
                f"Unknown VO parameter(s) {sorted(unknown)}. Valid: {sorted(tunable)}"
            )
        kw = {k: v for k, v in d.items() if k in tunable}
        kw.update(overrides)
        roles = kw.pop("d_pass_roles", None)
        cfg = cls(**kw)
        if isinstance(roles, dict):
            roles = RolePassDistances.from_dict(roles, cfg.d_pass, cfg.resolved_d_coll())
        cfg.d_pass_roles = roles or None
        return cfg


# Set per vessel at runtime, not from YAML.
_RUNTIME_FIELDS = {"nominal_psi", "ref_speed", "origin_north", "origin_east", "pref_vel_seq", "hazards"}


def make_vo_policy(
    cfg:                  VOAgentConfig,
    *,
    colregs_graph:        Optional[ColregsGraph] = None,
    own_vessel_idx:       int                    = 0,
    other_vessel_indices: Optional[list[int]]    = None,
) -> VOPolicyKuwata | VOPolicyAnalytic:
    """Build the VO policy selected by cfg.policy_type.

    cfg.nominal_psi and cfg.ref_speed must be set.  The LOS track-line
    origin is cfg.origin_north/east when both are set, else the vessel's
    position at the first select_action() call.
    """
    if cfg.nominal_psi is None or cfg.ref_speed is None:
        raise ValueError("make_vo_policy needs cfg.nominal_psi and cfg.ref_speed")
    origin_pos = (
        np.array([cfg.origin_north, cfg.origin_east], dtype=float)
        if cfg.origin_north is not None and cfg.origin_east is not None
        else None
    )
    common = dict(
        nominal_psi          = cfg.nominal_psi,
        ref_speed            = cfg.ref_speed,
        tcpa_threshold       = cfg.tcpa_threshold,
        d_pass               = cfg.d_pass,
        d_coll               = cfg.resolved_d_coll(),
        los_lookahead        = cfg.los_lookahead,
        head_on_rvo_weight   = cfg.head_on_rvo_weight,
        d_pass_roles         = cfg.d_pass_roles,
        origin_pos           = origin_pos,
        colregs_graph        = colregs_graph,
        own_vessel_idx       = own_vessel_idx,
        other_vessel_indices = other_vessel_indices,
    )
    if cfg.policy_type == "vo_analytic":
        return VOPolicyAnalytic(
            **common,
            continuity_weight = cfg.continuity_weight,
            course_budget_deg = cfg.course_budget_deg,
            achievable_ratio  = cfg.achievable_ratio,
            hazards           = cfg.hazards,
            terrain_lookahead_s = cfg.terrain_lookahead_s,
            terrain_clearance = cfg.terrain_clearance,
        )
    if cfg.policy_type == "vo_kuwata":
        return VOPolicyKuwata(
            **common,
            max_heading_change = cfg.max_heading_change,
            n_speeds           = cfg.n_speeds,
            n_headings         = cfg.n_headings,
        )
    raise ValueError(
        f"Unknown policy_type '{cfg.policy_type}'. Choose: 'vo_analytic' or 'vo_kuwata'."
    )


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def vo_rollout(
    initial_states:     dict[str, VesselState],
    dt:                 float,
    n_steps:            int,
    agent_configs:      Optional[dict[str, VOAgentConfig]]           = None,
    colregs_graph:      Optional[ColregsGraph]                       = None,
    vessel_index_map:   Optional[dict[str, int]]                     = None,
    colregs_classifier: Optional[ColregsClassifier]                  = None,
    intents:            Optional[dict[int, Intent]]                  = None,
    fixed_plan_ids:     Optional[set[str]]                           = None,
    fixed_tracks:       Optional[dict[str, list[VesselState]]]       = None,
    _timings:           Optional[dict]                               = None,
) -> dict[str, list[VesselState]]:
    """Predict multi-vessel trajectories assuming all vessels follow a VO policy.

    Parameters
    ----------
    initial_states : dict mapping vessel_id -> current VesselState.
        These are the t=0 states; the rollout does not modify them.
    dt : float
        Simulation timestep [s].
    n_steps : int
        Number of steps to roll out.  The returned trajectories have
        n_steps + 1 entries each (including the initial state).
    agent_configs : optional dict mapping vessel_id -> VOAgentConfig.
        Vessels without an entry use the default VOAgentConfig.
    colregs_graph : optional ColregsGraph shared across all vessels.
        When supplied each vessel's policy receives its own view of the
        graph (own_vessel_idx, other_vessel_indices) so COLREGs roles are
        respected during the rollout.  None → all roles default to
        AVOID_COLLISION (plain VO behaviour).  Ignored when
        colregs_classifier is provided.
    vessel_index_map : optional dict mapping vessel_id -> graph index.
        Required when colregs_graph or colregs_classifier is supplied.
        Maps vessel string IDs to the integer indices used by the graph
        (0 = ego, 1..N = targets).
    colregs_classifier : optional ColregsClassifier.
        When supplied, update() is called at every rollout step with the
        current states, and the resulting graph is pushed to all vessel
        policies via set_colregs_graph().  vessel_index_map must also be
        provided.  If colregs_graph is supplied too, it is taken as the
        already-classified graph for step 0 and the classifier only runs
        from step 1 onward (avoids a second hysteresis update on the t=0
        states the caller already classified).  The classifier is mutated.
    intents : optional dict mapping graph index -> Intent.
        Responsiveness assignment of this rollout.  Re-classified graphs are
        overridden with it (apply_responsiveness), and NON_RESPONSIVE vessels
        ignore all traffic and follow their nominal track.  None -> all
        vessels responsive.
    fixed_plan_ids : optional set of vessel ids that execute a given plan.
        These vessels ignore all traffic and track their pref_vel_seq (set it
        in agent_configs), while keeping their COLREGs roles, so the other
        vessels react to the plan as to a responsive vessel.  Used to
        evaluate own-ship manoeuvres.
    fixed_tracks : optional vessel id -> states at k * dt, k = 0..n_steps
        (at least that many). Like fixed_plan_ids, but the vessel replays
        the given track exactly instead of tracking a plan through its
        motion model. track[0] must be its initial state.

    Returns
    -------
    trajectories : dict[str, list[VesselState]]
        trajectories[vid][k] is the predicted state of vessel vid at
        time k * dt.  Length of each list is n_steps + 1.
    """
    if not initial_states:
        return {}

    _t0 = time.perf_counter() if _timings is not None else 0.0

    configs = agent_configs or {}
    default_cfg = VOAgentConfig()

    vessel_ids = list(initial_states.keys())

    # Build fresh policy + motion model for every vessel
    policies: dict[str, VOPolicyKuwata | VOPolicyAnalytic] = {}
    adapters: dict[str, BaseSimModel] = {}

    for vid, state in initial_states.items():
        cfg = configs.get(vid, default_cfg)

        nominal_psi = cfg.nominal_psi if cfg.nominal_psi is not None else state.psi
        ref_speed   = cfg.ref_speed   if cfg.ref_speed   is not None else state.speed

        if ref_speed <= 0.0:
            raise ValueError(
                f"Vessel '{vid}' has ref_speed={ref_speed} <= 0.  "
                "VO policy requires a positive reference speed."
            )

        own_idx    = vessel_index_map[vid] if vessel_index_map else 0
        other_idxs = (
            [vessel_index_map[other] for other in vessel_ids if other != vid]
            if vessel_index_map else None
        )

        policy = make_vo_policy(
            replace(cfg, nominal_psi=nominal_psi, ref_speed=ref_speed),
            colregs_graph        = colregs_graph,
            own_vessel_idx       = own_idx,
            other_vessel_indices = other_idxs,
        )
        if isinstance(policy, VOPolicyAnalytic):
            # Seed the continuity (hysteresis) memory with the vessel's current
            # velocity.  A fresh policy has _v_prev=None, so the first rollout
            # step would rank candidates with no memory while the simulated /
            # real vessel's persistent policy does — near-tied branches then
            # flip between frames and the rollout disagrees with the sim.
            policy._v_prev = np.array([state.vn, state.ve], dtype=float)
        policies[vid] = policy

        adapter = build_adapter(cfg.rollout_model, dt, k_psi=cfg.k_psi, tau_speed=cfg.tau_speed)
        adapter.init(InitialState(
            north   = state.north,
            east    = state.east,
            speed   = state.speed,
            heading = state.psi,
        ))
        adapters[vid] = adapter

    # Seed trajectories with t=0 states
    current:      dict[str, VesselState]       = dict(initial_states)
    trajectories: dict[str, list[VesselState]] = {
        vid: [s] for vid, s in initial_states.items()
    }

    # Pre-compute nominal positions by integrating pref_vel_seq from the
    # initial position.  Used as the LOS origin at each step so the policy
    # corrects back toward the planned route after a VO detour, rather than
    # following the sequence direction from wherever the vessel currently is.
    nom_positions: dict[str, list[np.ndarray]] = {}
    if configs:
        for vid, state in initial_states.items():
            cfg = configs.get(vid)
            if cfg is None or cfg.pref_vel_seq is None:
                continue
            pos = state.position.copy()
            pts: list[np.ndarray] = [pos.copy()]
            for h in range(n_steps):
                vp = cfg.pref_vel_seq[min(h, len(cfg.pref_vel_seq) - 1)]
                pos = pos + np.asarray(vp, dtype=float) * dt
                pts.append(pos.copy())
            nom_positions[vid] = pts

    if _timings is not None:
        _t1 = time.perf_counter(); _timings["rol.policy_setup"].append(_t1 - _t0); _t0 = _t1

    # A non-responsive vessel does not react to any traffic: it sees no other
    # vessels and follows its nominal track (same as NominalLOSPolicy in the
    # simulation).  Its STAND_ON role only tells the others it will not yield.
    def _non_responsive(i: int, vid: str) -> bool:
        idx = vessel_index_map[vid] if vessel_index_map else i
        return intents is not None and intents.get(idx) == Intent.NON_RESPONSIVE

    tracks = fixed_tracks or {}
    fixed = (fixed_plan_ids or set()) | set(tracks)
    other_ids: dict[str, list[str]] = {
        vid: ([] if _non_responsive(i, vid) or vid in fixed
              else [o for o in vessel_ids if o != vid])
        for i, vid in enumerate(vessel_ids)
    }

    for step_idx in range(n_steps):
        # Fixed-plan vessels whose plan has them (nearly) stopped this step:
        # the policies below only take preferred velocities above 0.1 m/s,
        # so the planned velocity is commanded directly instead.
        held: dict[str, CVAction] = {}

        # Step each vessel's policy along its nominal velocity sequence.
        # This overrides nominal_psi and ref_speed at every rollout step so the
        # VO solver tracks the full preferred-velocity trajectory, not just the
        # heading/speed frozen at rollout t=0.
        if configs:
            for vid in vessel_ids:
                cfg = configs.get(vid)
                if cfg is None or cfg.pref_vel_seq is None:
                    continue
                seq = cfg.pref_vel_seq
                h = min(step_idx, len(seq) - 1)
                vp = seq[h]
                spd = float(np.linalg.norm(vp))
                if spd > 0.1:
                    psi = float(math.atan2(float(vp[1]), float(vp[0])))
                    pol = policies[vid]
                    pol.nominal_psi  = psi
                    pol.ref_speed    = spd
                    pol._v_pref_unit = np.array([math.cos(psi), math.sin(psi)])
                    pol._v_pref      = np.array([vp[0], vp[1]], dtype=float)
                    pol._origin_pos  = nom_positions[vid][step_idx].copy()
                elif vid in fixed:
                    held[vid] = CVAction(vn=float(vp[0]), ve=float(vp[1]))

        # Update COLREGs graph from classifier if provided
        if (colregs_classifier is not None and vessel_index_map is not None
                and not (step_idx == 0 and colregs_graph is not None)):
            int_keyed = {vessel_index_map[vid]: current[vid] for vid in vessel_ids}
            new_graph = colregs_classifier.update(int_keyed)
            if intents is not None:
                new_graph = apply_responsiveness(new_graph, intents)
            for vid in vessel_ids:
                policies[vid].set_colregs_graph(new_graph)

        # All vessels observe the same snapshot before any of them acts
        _ts = time.perf_counter() if _timings is not None else 0.0
        actions: dict[str, CVAction] = {
            vid: held[vid] if vid in held else policies[vid].select_action(
                current[vid],
                [current[other] for other in other_ids[vid]],
            )
            for vid in vessel_ids if vid not in tracks
        }
        if _timings is not None:
            _ta = time.perf_counter(); _timings["rol.select_action"].append(_ta - _ts); _ts = _ta

        # Propagate all vessels forward with their chosen actions
        for vid in vessel_ids:
            new_state = (tracks[vid][step_idx + 1] if vid in tracks
                         else adapters[vid].step(actions[vid], dt))
            current[vid] = new_state
            trajectories[vid].append(new_state)

        if _timings is not None:
            _timings["rol.adapter_step"].append(time.perf_counter() - _ts)

    return trajectories


# ---------------------------------------------------------------------------
# Hypothesis enumeration + rollout
# ---------------------------------------------------------------------------

def rank_hypotheses(
    vessel_ids:       list[str],
    responding_probs: dict[str, float],
    top_k:            int,
) -> list[tuple[dict[str, Intent], float]]:
    """The top_k most likely responsiveness assignments over vessel_ids.

    Each vessel is independently responsive with p = responding_probs[vid]
    (missing -> 1.0). Returns (assignment, probability) pairs, most likely
    first; zero-probability assignments are dropped.
    """
    hypotheses: list[tuple[tuple[Intent, ...], float]] = []
    for assignment in itertools_product([Intent.RESPONSIVE, Intent.NON_RESPONSIVE],
                                        repeat=len(vessel_ids)):
        prob = 1.0
        for vid, intent in zip(vessel_ids, assignment):
            p_r = responding_probs.get(vid, 1.0)
            prob *= p_r if intent == Intent.RESPONSIVE else (1.0 - p_r)
        hypotheses.append((assignment, prob))

    hypotheses.sort(key=lambda x: x[1], reverse=True)
    return [
        (dict(zip(vessel_ids, assignment)), prob)
        for assignment, prob in hypotheses[:top_k]
        if prob >= 1e-9
    ]


def enumerate_and_rollout(
    initial_states:   dict[str, VesselState],
    responding_probs: dict[str, float],
    nom_graph:        ColregsGraph,
    dt:               float,
    n_steps:          int,
    agent_configs:    Optional[dict[str, VOAgentConfig]]                    = None,
    vessel_index_map: Optional[dict[str, int]]                              = None,
    top_k:            int                                                   = 1,
    colregs_classifier: Optional[ColregsClassifier]                         = None,
    fixed_plan_ids:   Optional[set[str]]                                    = None,
    pool:             Optional["RolloutPool"]                               = None,
    _timings:         Optional[dict]                                        = None,
) -> list[tuple[dict[str, list[VesselState]], dict[str, Intent], float]]:
    """Enumerate 2^M COLREGs graph hypotheses from per-vessel responsiveness posteriors
    and realize the top-K as trajectory bundles via vo_rollout.

    Parameters
    ----------
    responding_probs : p(responsive) per vessel_id. Vessels not in the dict default to 1.0.
    nom_graph : G_nom from the COLREGs classifier (before apply_responsiveness).
    top_k : maximum number of hypotheses to realize.
    colregs_classifier : optional classifier whose state already reflects
        nom_graph at t (i.e. update() was called on initial_states).  When
        given, each hypothesis rollout re-classifies at every step h >= 1 on
        a private deep copy and overrides the result with that hypothesis's
        intents (paper Sec. V-D).  None -> the graph is held fixed at
        effective_graph for the whole horizon.  The caller's classifier is
        never mutated.
    fixed_plan_ids : vessels that execute the pref_vel_seq in their agent
        config regardless of traffic (see vo_rollout).
    pool : optional RolloutPool (prediction/parallel.py).  Given, the
        hypothesis rollouts run in parallel worker processes; results are
        identical to the serial run.  _timings is then not collected.

    Returns
    -------
    list of (trajectories, assignment, weight) sorted by weight descending,
    length <= top_k.  ``assignment`` maps vessel_id -> Intent for that hypothesis.
    """
    return start_enumerate_and_rollout(
        initial_states, responding_probs, nom_graph, dt, n_steps,
        agent_configs      = agent_configs,
        vessel_index_map   = vessel_index_map,
        top_k              = top_k,
        colregs_classifier = colregs_classifier,
        fixed_plan_ids     = fixed_plan_ids,
        pool               = pool,
        _timings           = _timings,
    ).result()


def start_enumerate_and_rollout(
    initial_states:   dict[str, VesselState],
    responding_probs: dict[str, float],
    nom_graph:        ColregsGraph,
    dt:               float,
    n_steps:          int,
    agent_configs:    Optional[dict[str, VOAgentConfig]]                    = None,
    vessel_index_map: Optional[dict[str, int]]                              = None,
    top_k:            int                                                   = 1,
    colregs_classifier: Optional[ColregsClassifier]                         = None,
    fixed_plan_ids:   Optional[set[str]]                                    = None,
    pool:             Optional["RolloutPool"]                               = None,
    _timings:         Optional[dict]                                        = None,
) -> "Pending":
    """enumerate_and_rollout, split in two: the rollouts are started (on the
    pool, if given) when this returns, and .result() on the returned Pending
    waits for them and gives enumerate_and_rollout's return value.

    Lets a caller start the rollouts of one frame and go on with the next
    frame while they run (see scripts/run_AIS.py, metrics_run).  Without a
    pool the rollouts run here, before this returns.
    """
    from colregs_planner.prediction.parallel import Pending, submit_rollout

    vessel_ids = list(initial_states.keys())

    # Start one rollout per hypothesis.
    started: list[tuple[dict[str, Intent], float, "Pending"]] = []
    for assignment_by_vid, weight in rank_hypotheses(vessel_ids, responding_probs, top_k):
        intents: dict[int, Intent] = {
            (vessel_index_map[vid] if vessel_index_map else i): assignment_by_vid[vid]
            for i, vid in enumerate(vessel_ids)
        }
        effective_graph = apply_responsiveness(nom_graph, intents)
        handle = submit_rollout(
            pool,
            initial_states   = initial_states,
            dt               = dt,
            n_steps          = n_steps,
            agent_configs    = agent_configs,
            colregs_graph    = effective_graph,
            vessel_index_map = vessel_index_map,
            colregs_classifier = (copy.deepcopy(colregs_classifier)
                                  if colregs_classifier is not None else None),
            intents          = intents,
            fixed_plan_ids   = fixed_plan_ids,
            _timings         = _timings,
        )
        started.append((assignment_by_vid, weight, handle))

    # Collect them in the same order.
    def collect() -> list[tuple[dict[str, list[VesselState]], dict[str, Intent], float]]:
        results = []
        for assignment_by_vid, weight, handle in started:
            try:
                trajs = handle.result()
            except ValueError:
                trajs = {vid: [s] * (n_steps + 1) for vid, s in initial_states.items()}
            results.append((trajs, assignment_by_vid, weight))
        return results

    return Pending(collect)


# ---------------------------------------------------------------------------
# Nominal 1-step prediction (no VO avoidance)
# ---------------------------------------------------------------------------

def nominal_1step(
    state: VesselState,
    cfg:   Optional[VOAgentConfig],
    dt:    float,
) -> np.ndarray:
    """Return a 1-step nominal state prediction with LOS guidance, no VO.

    Uses the desired nominal velocity directly (no heading-rate dynamics) so
    the NC prediction represents where the vessel *should be* on the nominal
    track, not where it is currently heading. This keeps the two hypotheses
    distinguishable when the vessel has maneuvered away from its nominal course.

    Desired velocity priority:
      1. pref_vel_seq[0]       — library-retrieved nominal velocity (no LOS)
      2. nominal_psi/ref_speed — LOS-corrected heading toward the original track
      3. current velocity       — fallback when no config is available
    """
    if cfg is not None and cfg.pref_vel_seq is not None and len(cfg.pref_vel_seq) > 0:
        vp = np.asarray(cfg.pref_vel_seq[0], dtype=float)
    elif cfg is not None and cfg.nominal_psi is not None:
        spd = cfg.ref_speed if cfg.ref_speed is not None else state.speed
        psi = cfg.nominal_psi
        if cfg.origin_north is not None and cfg.origin_east is not None:
            # Apply LOS cross-track correction (same formula as VOPolicyAnalytic).
            cn, sn = math.cos(psi), math.sin(psi)
            dx = state.north - cfg.origin_north
            dy = state.east  - cfg.origin_east
            e  = dx * sn - dy * cn          # signed cross-track error
            psi = psi + math.atan2(e, cfg.los_lookahead)
        vp = np.array([spd * math.cos(psi), spd * math.sin(psi)])
    else:
        vp = np.array([state.vn, state.ve])
    return np.array([state.north + vp[0] * dt, state.east + vp[1] * dt, vp[0], vp[1]])

