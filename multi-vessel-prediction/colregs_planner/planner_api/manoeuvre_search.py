"""planner_api/manoeuvre_search.py

Receding-horizon search over sequences of discrete own-ship manoeuvres,
scored with the interaction-aware predictor.

A plan is a sequence of ``depth`` edges of ``edge_s`` seconds each. On every
edge the own ship sails at a course offset from its nominal course (and a
speed factor), then holds its nominal course and speed for the rest of the
horizon, which leaves it on a parallel offset track. Each plan is rolled out
with the own ship executing it (vo_rollout's fixed-plan mode, through the
prediction model) while every other vessel reacts to it, under each of the
top-K responsiveness hypotheses, and the plan with the lowest cost is chosen.

With depth 2 the tree is small enough to search exhaustively, so the result
is the optimum over the action set; no heuristic or pruning is involved.

Cost of a plan, per hypothesis h with renormalised weight w_h:

    safety_h   = sum_j  w_safe * max(short_j, bow_j)^2 + w_coll * [d_j < d_coll]
      short_j  = max(0, d_pass - d_j) / d_pass
      bow_j    = max(0, B - bcr_j) / B,   B = bow_factor * d_pass
    colregs_h  = w_port     * [any port offset while give-way/head-on to a vessel]
               + w_give_way * [first edge does not alter to starboard while
                               head-on to a vessel it is in conflict with
                               (Rule 14), or keeps course and speed while
                               give-way to a vessel it is in conflict with
                               (Rule 16: act early)]
               + w_stand_on * [manoeuvre on the first edge while stand-on to a
                               vessel that is responsive under h, and not
                               bound to keep clear of any vessel]   (Rule 17(a)(i))
               + w_give_way * [first edge does not alter course away from
                               a vessel being overtaken, while in conflict
                               with it]   (Rules 13, 16)
               + w_port     * [any edge alters course toward a vessel being
                               overtaken in range]   (Rule 13: keep clear
                               until finally past and clear)

A vessel being overtaken is in range when, on the own ship's route course,
the TCPA is below the predictor's tcpa_threshold. It is in conflict when
the DCPA is also below the overtaking passing distance (d_pass_roles, else
d_pass). Until a vessel being overtaken is in range, it only costs a
collision (d_j < d_coll), not a passing distance short of d_pass. Otherwise
the horizon would see the conflict first, and the search would slow down
behind the vessel instead of passing. The collision cost stays because
"in range" is judged on the route course, while a plan may turn toward the
vessel. Once it is in range, a conflict triggers the
COLREGs term, which makes the search commit to altering course. d_j then
also includes the straight-line CPA from the end of the rollout. Overtaking
closes slowly, so the pass often lies beyond the horizon, and a plan that
drifts back onto the route before passing would otherwise look safe.

"Away" and "toward" depend on which side of the overtaken vessel's track
the own ship is on: on its starboard side, starboard is away. Dead astern
(within 1 m of its track), either way is away. Without the toward term, the
own ship sits near d_pass, where the conflict flickers on and off. Each time
it reappears, a port turn back toward the route also escapes the
alter-course term while costing less deviation, so the ship wiggles back
and forth while overtaking.

A give-way or head-on vessel is in conflict while resuming the route course
from the current position would pass it inside that role's passing distance
(d_pass_roles, else d_pass). Once the manoeuvre has opened that distance,
holding course is no longer penalised. Otherwise the own ship keeps
"acting", for instance slowing down, until every such vessel is past. The
ban on port turns toward a give-way or head-on vessel still holds for the
whole encounter.

The stand-on term lapses while the own ship must keep clear of another
approaching vessel (a give-way, head-on or overtaking conflict). Rule 17's
duty to keep course and speed holds toward the vessel that keeps clear; it
cannot oblige a ship to stand on into a vessel it must itself avoid. With
both terms, holding course is the cheaper violation (w_give_way <
w_stand_on), and the search postpones the give-way manoeuvre to "the next
edge" at every replan. The safety and port terms still keep the manoeuvre
clear of the stand-on vessel.

With pass_margin > 0, d_pass in safety_h (short_j, bow_j and the
overtaking extension) is d_pass + pass_margin: a buffer for prediction
error, since a plan that only just clears d_pass costs nothing. The role
conflicts and the deviation scale keep d_pass.

d_j is the closest approach to vessel j over the rollout. Crossing ahead of a
vessel needs more room than passing astern of it (Rule 15: the give-way
vessel avoids crossing ahead). bcr_j, the bow crossing range, is how far
ahead of vessel j the own ship is when it crosses j's heading line, at the
nearest such crossing in the rollout; without a crossing ahead, bow_j is 0.
It is about crossing the track, not about being ahead of the bow: in a
head-on approach each ship is ahead of the other's bow throughout, without
crossing ahead. The distance at a crossing is at least d_j, so with
bow_factor = 1 the bow term never adds anything.

The worst case only ranges over hypotheses with weight >= risk_floor. Over
all of them, a 1 % "the give-way vessel never acts" hypothesis would carry a
collision cost that outweighs every COLREGs term, and a stand-on own ship
would manoeuvre from the first step instead of holding course until the
filter actually doubts the other vessel (Rule 17).

Roles only count toward vessels still approaching (TCPA > 0): the classifier
keeps an encounter's roles until the vessels have opened to d_u, and once
they are past each other the rules no longer bind the return to track. For
a vessel being overtaken, the TCPA is taken on the route course, so it is
past once drawn level: turning away opens the range on the actual velocity
long before that.
    cost = sum_h w_h (safety_h + colregs_h) + risk_weight * max_{h: w_h >= risk_floor} safety_h
         + w_dev * mean_t (cross-track offset(t) / d_pass)^2
         + w_effort * sum_edges (|offset| / 60 deg + [slowed or stopped]) * edge_s / effort_ref_s
         + w_time * (time lost along the route over the horizon) / time_ref_s
         + w_ground * max(0, (d_shore - g) / d_shore)^2 + w_aground * [g < 0]

where d_j is the closest approach to vessel j over the horizon. The
cross-track term is averaged over the horizon rather than taken at its end:
an end-only term ties "return now" with "return on the next edge", and
replanning then postpones the return indefinitely.

Hull response: the own ship's track along a plan is also the reference
its DP controller follows (policies/search_dp.py), and the hull lags it: a
stop takes about 8 m past where the reference stops. With a HullModel set,
the plan is scored, and the other vessels react to it, along the hull's
predicted track instead of the reference. The reference itself is
unchanged, so DP still brakes as hard as it can.

Terrain: with a HazardMap (seachart/hazard.py) set, g is the own ship's
closest approach to land or shoal over the horizon, along the same track
the plan is scored on (the hull's, with a HullModel). Both terms are 0
without a map. Like safety, the terrain cost counts toward "less safe" in
the hysteresis, so a plan in force that has turned toward the bank is
dropped. The other vessels do not see the terrain yet.

Hysteresis: the plan in force is continued edge by edge. Adopted at t0,
at time t its continuation is its actions from edge k = floor((t - t0) /
edge_s) on, padded with nominal, each edge at full length: (stop, stbd_60)
adopted at 0 continues as itself until 7.5 s, then (stbd_60, nominal),
then (nominal, nominal). That is one of the grid plans, so it is scored
anyway, and it is followed unless the cheapest plan is cheaper by more
than max(hold_abs_margin, hold_rel_margin * its cost). The limit is
safety: the continuation is dropped as soon as its CPA cost (safety +
risk + ground) exceeds the cheapest plan's by more than hold_safety_tol, whatever
the margin. The margin is what still lets the ship return to its route.
Edges are re-anchored at full length rather than shortened by the time
already sailed: a plan relies on replanning to extend it, so its literal
remainder resumes nominal too early and is always less safe.
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from colregs_planner.colregs_graph.colregs_classifier import ColregsClassifier, _compute_dcpa_tcpa
from colregs_planner.colregs_graph.colregs_graph import ColregsGraph, VesselRole, apply_responsiveness
from colregs_planner.prediction.parallel import RolloutPool, submit_rollout
from colregs_planner.prediction.vo_rollout import VOAgentConfig, rank_hypotheses
from colregs_planner.seachart.hazard import HazardMap
from colregs_planner.vessel.state import Intent, VesselState, wrap_angle


@dataclass(frozen=True)
class SearchAction:
    """One edge of a plan: course offset from nominal [deg, + = starboard] and speed factor."""

    name:          str
    course_offset: float = 0.0
    speed_factor:  float = 1.0

    @property
    def is_nominal(self) -> bool:
        return self.course_offset == 0.0 and self.speed_factor == 1.0


# Alterations large enough to be readily apparent (Rule 8(b)), starboard
# preferred; port and slowing down are available but carry costs.
DEFAULT_ACTIONS: tuple[SearchAction, ...] = (
    SearchAction("nominal"),
    SearchAction("stbd_30", course_offset=30.0),
    SearchAction("stbd_60", course_offset=60.0),
    SearchAction("port_30", course_offset=-30.0),
    SearchAction("slow",    speed_factor=0.5),
    SearchAction("stop",    speed_factor=0.0),     # take all way off (Rule 8(e))
)


@dataclass
class SearchWeights:
    w_safe:      float = 100.0
    bow_factor:  float = 2.0    # required range when crossing ahead of a vessel, in d_pass
    w_coll:      float = 1000.0
    w_port:      float = 20.0
    w_give_way:  float = 5.0
    w_stand_on:  float = 10.0
    w_dev:       float = 2.0
    w_effort:    float = 1.0
    w_time:      float = 1.0
    time_ref_s:  float = 15.0   # time lost is charged per this many seconds, whatever edge_s is
    effort_ref_s: float = 15.0  # an edge's effort counts in full when held this long, whatever edge_s is
    risk_weight: float = 1.0
    risk_floor:  float = 0.1    # hypotheses below this weight stay out of the worst case
    pass_margin: float = 0.0    # [m] the safety cost aims for d_pass + pass_margin (see docstring)
    # Terrain (with SearchConfig.hazards): keep d_shore from land and shoals.
    w_ground:    float = 100.0
    w_aground:   float = 1000.0
    d_shore:     float = 8.0    # [m] from the ship's centre: half the beam plus the hull model's error
    # Hysteresis (see the module docstring): keep the previous plan unless a
    # fresh one is this much cheaper, or the previous one is less safe.
    hold_abs_margin: float = 0.1
    hold_rel_margin: float = 0.15
    hold_safety_tol: float = 0.0


# Cost terms (ScoredPlan.terms) grouped for display, each group with the
# SearchWeights fields that scale it. Every term is in exactly one group.
COST_CATEGORIES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "CPA":     (("safety", "risk"),     ("w_safe", "w_coll", "bow_factor", "risk_weight")),
    "COLREGs": (("colregs",),           ("w_port", "w_give_way", "w_stand_on")),
    "Route":   (("deviation", "time"),  ("w_dev", "w_time")),
    "Effort":  (("effort",),            ("w_effort",)),
    "Terrain": (("ground",),            ("w_ground", "w_aground")),
}


def cost_by_category(terms: dict[str, float]) -> dict[str, float]:
    """A plan's cost terms summed per COST_CATEGORIES group."""
    return {cat: sum(terms.get(t, 0.0) for t in ts) for cat, (ts, _) in COST_CATEGORIES.items()}


@dataclass
class HullModel:
    """Closed-loop response of the hull to a DP reference moving along a plan:

        a = kp (p_r - p) + kd (v_r - v) - c v,   |a| <= a_max

    integrated at h, with the reference moving linearly between its samples.
    Every reanchor_s the reference is moved onto the hull, as SearchDPPolicy
    does at each replan: the hull then does not back up onto a stop point it
    overshot, but creeps on from where it is. The defaults are fitted to the
    milliAmpere simulator's DP following the planner's own-ship tracks from
    0-1.5 m/s, re-anchored every 5 s (scripts/tools/fit_hull_model.py; RMS
    error 1.4 m over 40 s, worst 3.5 m, against 4.5 m and 10 m taking the
    reference as the hull). With that
    re-anchoring the position feedback fits to zero: the hull follows the
    reference velocity at rate kd, at most a_max."""

    kp:    float = 0.0
    kd:    float = 0.2016
    c:     float = 0.0
    a_max: float = 0.1706
    h:     float = 0.1
    reanchor_s: Optional[float] = None   # the own ship's replan interval

    def track(self, refs: np.ndarray, v0: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Hull positions and velocities (P, n+1, 2) along the reference
        positions *refs* (P, n+1, 2) at spacing dt, starting on the reference
        with velocity v0 [vn, ve]."""
        n_sub = max(1, int(round(dt / self.h)))
        h = dt / n_sub
        every = int(round(self.reanchor_s / dt)) if self.reanchor_s else 0
        refs = np.array(refs, dtype=float)
        vr = np.diff(refs, axis=1) / dt
        p = refs[:, 0].copy()
        v = np.tile(np.asarray(v0, dtype=float), (len(refs), 1))
        pos, vel = [p.copy()], [v.copy()]
        for k in range(refs.shape[1] - 1):
            if every and k and k % every == 0:
                refs[:, k:] += (p - refs[:, k])[:, None]
            for j in range(n_sub):
                a = (self.kp * (refs[:, k] + vr[:, k] * (j * h) - p)
                     + self.kd * (vr[:, k] - v) - self.c * v)
                na = np.linalg.norm(a, axis=1, keepdims=True)
                a *= np.minimum(1.0, self.a_max / np.maximum(na, 1e-12))
                v = v + a * h
                p = p + v * h
            pos.append(p.copy())
            vel.append(v.copy())
        return np.stack(pos, axis=1), np.stack(vel, axis=1)


@dataclass
class SearchConfig:
    depth:     int   = 2
    edge_s:    float = 7.5      # duration of each edge [s]
    plan_dt:   float = 1.0      # rollout timestep [s]
    horizon_s: float = 90.0     # rollout horizon [s]
    top_k:     int   = 2        # responsiveness hypotheses scored
    actions:   tuple[SearchAction, ...] = DEFAULT_ACTIONS
    weights:   SearchWeights = field(default_factory=SearchWeights)
    hull:      Optional[HullModel] = None   # score along the hull's track (see docstring)
    hazards:   Optional[HazardMap] = None   # land and shoals (see docstring)


@dataclass
class ScoredPlan:
    actions:   tuple[SearchAction, ...]
    cost:      float
    min_dist:  dict[str, float]          # worst case over hypotheses, per other vessel (see d_j above)
    terms:     dict[str, float]
    min_shore: float = math.inf          # closest approach to a hazard (g above) [m]


@dataclass
class SearchResult:
    best:          ScoredPlan
    ranked:        list[ScoredPlan]       # all plans, cheapest first
    hypotheses:    list[tuple[dict[str, Intent], float]]
    own_traj:      list[VesselState]      # own ship along the best plan, plan_dt apart (the DP reference)
    rollouts:      list[dict[str, list[VesselState]]]   # best plan, one per hypothesis
    plan_dt:       float
    kept:          Optional[ScoredPlan] = None   # continuation of the plan in force (in ranked)
    hull_traj:     Optional[list[VesselState]] = None   # predicted hull along own_traj (with a HullModel)


class ManoeuvreSearchPlanner:
    """Exhaustive search over manoeuvre sequences for one own ship.

    The own ship is the vessel under our control, so it always follows
    COLREGs: every hypothesis scored has it responsive.

    Parameters
    ----------
    own_id : the planning vessel.
    agent_configs : VOAgentConfig per vessel, as used by the predictor. The
        own ship's nominal_psi, ref_speed and origin define its route.
    vessel_index_map : vessel_id -> COLREGs graph index.
    pool : optional RolloutPool (prediction/parallel.py). Given, every plan's
        rollouts are submitted to it before any is scored, so they run in
        parallel. The result is the same either way.
    """

    def __init__(
        self,
        own_id:           str,
        agent_configs:    dict[str, VOAgentConfig],
        vessel_index_map: dict[str, int],
        config:           SearchConfig | None = None,
        pool:             RolloutPool | None = None,
    ) -> None:
        self.own_id    = own_id
        self.configs   = agent_configs
        self.index_map = vessel_index_map
        self.cfg       = config or SearchConfig()
        self.pool      = pool
        # The plan in force and when it was adopted: continued at the next
        # replan unless another plan clearly beats it (hysteresis).
        self._kept: Optional[tuple[tuple[SearchAction, ...], float]] = None

    def reset(self) -> None:
        """Forget the plan in force: the next plan() chooses freely."""
        self._kept = None

    # ------------------------------------------------------------------
    # Plans as velocity sequences
    # ------------------------------------------------------------------

    def _nominal_velocity(self) -> np.ndarray:
        own = self.configs[self.own_id]
        return own.ref_speed * np.array([math.cos(own.nominal_psi), math.sin(own.nominal_psi)])

    def velocity_sequence(self, actions: tuple[SearchAction, ...]) -> np.ndarray:
        """(H, 2) planned [vn, ve] at plan_dt for this action sequence."""
        c = self.cfg
        H = int(round(c.horizon_s / c.plan_dt))
        per_edge = int(round(c.edge_s / c.plan_dt))
        nominal = self._nominal_velocity()
        seq = np.tile(nominal, (H, 1))
        for i, a in enumerate(actions):
            rot = math.radians(a.course_offset)
            R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
            seq[i * per_edge:(i + 1) * per_edge] = (R @ nominal) * a.speed_factor
        return seq

    def _continuation(self, t: float) -> Optional[tuple[SearchAction, ...]]:
        """The plan in force continued at t: its edges from the one under way
        on, padded with nominal, each at full length. None before the first
        plan, or without a nominal action to pad with."""
        if self._kept is None:
            return None
        actions, t0 = self._kept
        nominal = next((a for a in self.cfg.actions if a.is_nominal), None)
        if nominal is None:
            return None
        k = min(len(actions), int(math.floor((t - t0) / self.cfg.edge_s + 1e-9)))
        return actions[k:] + (nominal,) * k

    def _sequences(self) -> list[tuple[SearchAction, ...]]:
        return list(itertools.product(self.cfg.actions, repeat=self.cfg.depth))

    def _bow_crossing_shortfall(self, own: np.ndarray, other: list[VesselState],
                                d_pass: float) -> float:
        """Shortfall of the bow crossing range from bow_factor * d_pass, as a
        fraction of it: the own ship (positions *own*) crossing *other*'s
        heading line ahead of it. 0 without such a crossing."""
        f = self.cfg.weights.bow_factor
        if f <= 1.0:
            return 0.0
        psi = np.array([s.psi for s in other])
        rel_n = own[:, 0] - np.array([s.north for s in other])
        rel_e = own[:, 1] - np.array([s.east for s in other])
        along = rel_n * np.cos(psi) + rel_e * np.sin(psi)       # ahead of the other vessel
        lateral = rel_e * np.cos(psi) - rel_n * np.sin(psi)     # to its starboard
        under_way = np.array([s.speed for s in other])[:-1] >= 0.1
        side = lateral >= 0.0          # a sample exactly on the line counts as starboard
        cross = np.nonzero((side[:-1] != side[1:]) & under_way)[0]
        if cross.size == 0:
            return 0.0
        frac = lateral[cross] / (lateral[cross] - lateral[cross + 1])
        at = along[cross] + frac * (along[cross + 1] - along[cross])
        ahead = at[at > 0.0]
        if ahead.size == 0:
            return 0.0
        required = f * d_pass
        return max(0.0, (required - float(ahead.min())) / required)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def plan(
        self,
        states:           dict[str, VesselState],
        nom_graph:        ColregsGraph,
        t:                float,
        responding_probs: Optional[dict[str, float]] = None,
        intents:          Optional[dict[str, Intent]] = None,
        classifier:       Optional[ColregsClassifier] = None,
    ) -> SearchResult:
        """Choose the best plan from the current situation.

        responding_probs : p(responsive) per vessel from the filters; the top_k
            hypotheses are scored. Without it, *intents* (or all responsive)
            is the single hypothesis. Either way the own ship is responsive.
        classifier : if given, each rollout re-classifies on a private copy.
        """
        c, w = self.cfg, self.cfg.weights
        H = int(round(c.horizon_s / c.plan_dt))
        ids = list(states)

        if responding_probs is not None:
            probs = dict(responding_probs)
            probs[self.own_id] = 1.0
            hyps = rank_hypotheses(ids, probs, c.top_k)
        else:
            base = {**(intents or {}), self.own_id: Intent.RESPONSIVE}
            hyps = [({vid: base.get(vid, Intent.RESPONSIVE) for vid in ids}, 1.0)]
        total = sum(p for _, p in hyps)
        hyps = [(a, p / total) for a, p in hyps]

        own_cfg = self.configs[self.own_id]
        d_pass, d_coll = own_cfg.d_pass, own_cfg.resolved_d_coll()
        d_safe = d_pass + w.pass_margin
        own_idx = self.index_map[self.own_id]
        idx_to_vid = {i: vid for vid, i in self.index_map.items()}

        # Ego's roles per hypothesis, from the graph at t.
        graphs = []
        for assignment, _ in hyps:
            int_intents = {self.index_map[vid]: assignment[vid] for vid in ids}
            graphs.append((int_intents, apply_responsiveness(nom_graph, int_intents)))

        others = [vid for vid in ids if vid != self.own_id]
        approaching = {
            vid for vid in others
            if _compute_dcpa_tcpa(states[self.own_id], states[vid])[1] > 0
        }
        own_now = states[self.own_id]
        on_route = VesselState(own_now.north, own_now.east, own_cfg.nominal_psi, own_cfg.ref_speed)
        # Give-way and head-on encounters still to resolve: resuming the route
        # course from here would pass inside that role's passing distance.
        roles_d = own_cfg.d_pass_roles
        d_give_way = roles_d.give_way if roles_d else d_pass
        d_head_on = roles_d.head_on if roles_d else d_pass
        route_dcpa = {vid: _compute_dcpa_tcpa(on_route, states[vid])[0] for vid in approaching}
        give_way_conflict = {vid for vid, d in route_dcpa.items() if d < d_give_way}
        head_on_conflict = {vid for vid, d in route_dcpa.items() if d < d_head_on}
        # Vessels the own ship is overtaking (Rule 13), within the predictor's
        # TCPA threshold on the route course. The duty to keep clear does not
        # depend on whether the other vessel responds, so this uses the
        # nominal graph rather than each hypothesis' graph. In conflict:
        # resuming the route course from here would pass inside the
        # overtaking passing distance.
        d_overtake = own_cfg.d_pass_roles.overtaking if own_cfg.d_pass_roles else d_pass
        overtaken: set[str] = set()         # in range
        overtaken_far: set[str] = set()     # not yet in range: collision cost only
        overtake_conflict = False
        # +1: own ship on the starboard side of the overtaken vessel's track
        # (starboard alterations are away from it), -1: port side, 0: astern.
        overtake_side: dict[str, int] = {}
        for j, r in nom_graph.get_roles(own_idx):
            vid = idx_to_vid.get(j)
            if r != VesselRole.OVERTAKING or vid not in others:
                continue
            # Past once drawn level, judged on the route course: while the
            # own ship is altering course away, the TCPA on its actual
            # velocity can turn negative long before it has passed.
            dcpa, tcpa = _compute_dcpa_tcpa(on_route, states[vid])
            if tcpa <= 0.0:
                continue
            if tcpa < own_cfg.tcpa_threshold:
                overtaken.add(vid)
                overtake_conflict |= dcpa < d_overtake
                tgt = states[vid]
                lateral = (math.cos(tgt.psi) * (own_now.east - tgt.east)
                           - math.sin(tgt.psi) * (own_now.north - tgt.north))
                overtake_side[vid] = 0 if abs(lateral) < 1.0 else (1 if lateral > 0 else -1)
            else:
                overtaken_far.add(vid)
        scored: list[tuple[ScoredPlan, list[dict[str, list[VesselState]]], Optional[list[VesselState]]]] = []

        # Submit every rollout first, so that with a pool they all run in parallel.
        sequences = self._sequences()
        # The own ship sails each plan as planned, without its VO's terrain
        # check: that would bend the plan away from the banks, and the
        # terrain cost below judges the plan itself.
        own_cfgs = [replace(own_cfg, pref_vel_seq=self.velocity_sequence(a), hazards=None)
                    for a in sequences]
        refs = hulls = None
        if c.hull is not None:
            # The own ship ignores traffic, so its reference is the same alone.
            refs = [h.result()[self.own_id] for h in [
                submit_rollout(self.pool, initial_states={self.own_id: own_now}, dt=c.plan_dt,
                               n_steps=H, agent_configs={self.own_id: cfg},
                               fixed_plan_ids={self.own_id})
                for cfg in own_cfgs
            ]]
            pos, vel = c.hull.track(np.array([[[s.north, s.east] for s in r] for r in refs]),
                                    [own_now.vn, own_now.ve], c.plan_dt)
            hulls = [_states(p, v, own_now.psi) for p, v in zip(pos, vel)]
        pending = []
        for i, actions in enumerate(sequences):
            configs = dict(self.configs)
            configs[self.own_id] = own_cfgs[i]
            pending.append([
                submit_rollout(
                    self.pool,
                    initial_states     = states,
                    dt                 = c.plan_dt,
                    n_steps            = H,
                    agent_configs      = configs,
                    colregs_graph      = graph,
                    vessel_index_map   = self.index_map,
                    colregs_classifier = copy.deepcopy(classifier) if classifier else None,
                    intents            = int_intents,
                    fixed_plan_ids     = {self.own_id},
                    fixed_tracks       = {self.own_id: hulls[i]} if hulls else None,
                )
                for int_intents, graph in graphs
            ])

        for i, (actions, handles) in enumerate(zip(sequences, pending)):
            safety, colregs, worst_d = [], [], {vid: math.inf for vid in others}
            rollouts = []
            for (assignment, _), (int_intents, graph), handle in zip(hyps, graphs, handles):
                trajs = handle.result()
                rollouts.append(trajs)
                own = np.array([[s.north, s.east] for s in trajs[self.own_id]])
                s_h = 0.0
                for vid in others:
                    tr = trajs[vid]
                    d = float(np.min(np.linalg.norm(
                        own - np.array([[s.north, s.east] for s in tr]), axis=1)))
                    shortfall = (d_safe - d) / d_safe
                    if vid in overtaken:
                        d_ext, t_ext = _compute_dcpa_tcpa(trajs[self.own_id][-1], trajs[vid][-1])
                        if t_ext > 0:
                            d = min(d, d_ext)
                            shortfall = max(shortfall, (d_safe - d_ext) / d_safe)
                    worst_d[vid] = min(worst_d[vid], d)
                    if vid not in overtaken_far:
                        shortfall = max(shortfall, self._bow_crossing_shortfall(own, tr, d_safe))
                        s_h += w.w_safe * max(0.0, shortfall) ** 2
                    s_h += w.w_coll * (d < d_coll)
                safety.append(s_h)

                c_h = 0.0
                roles = {idx_to_vid[j]: r for j, r in graph.get_roles(own_idx)
                         if idx_to_vid[j] in approaching}
                head_on = any(r == VesselRole.HEAD_ON and assignment.get(vid) == Intent.RESPONSIVE
                              for vid, r in roles.items())
                give_way = VesselRole.GIVE_WAY in roles.values()
                give_way_open = any(r == VesselRole.GIVE_WAY and vid in give_way_conflict
                                    for vid, r in roles.items())
                head_on_open = any(r == VesselRole.HEAD_ON and vid in head_on_conflict
                                   and assignment.get(vid) == Intent.RESPONSIVE
                                   for vid, r in roles.items())
                if (head_on or give_way) and any(a.course_offset < 0 for a in actions):
                    c_h += w.w_port
                if ((head_on_open and actions[0].course_offset <= 0)
                        or (give_way_open and actions[0].is_nominal)):
                    c_h += w.w_give_way
                if overtake_conflict and not any(
                    actions[0].course_offset * side > 0 or (side == 0 and actions[0].course_offset)
                    for side in overtake_side.values()
                ):
                    c_h += w.w_give_way
                if any(a.course_offset * side < 0
                       for a in actions for side in overtake_side.values()):
                    c_h += w.w_port
                # Stand on only when not bound to keep clear of another vessel.
                if not (head_on_open or give_way_open or overtake_conflict) and not actions[0].is_nominal and any(
                    r == VesselRole.STAND_ON and assignment.get(vid) == Intent.RESPONSIVE
                    for vid, r in roles.items()
                ):
                    c_h += w.w_stand_on
                colregs.append(c_h)

            # Cross-track offset from the nominal route over the horizon.
            own_tr = rollouts[0][self.own_id]
            psi0 = own_cfg.nominal_psi
            xtrack = np.array([
                (s.east - own_cfg.origin_east) * math.cos(psi0)
                - (s.north - own_cfg.origin_north) * math.sin(psi0)
                for s in own_tr
            ])
            # Per edge, in proportion to how long it is held, so the effort of
            # an alteration does not depend on the step length.
            effort = (sum(abs(a.course_offset) / 60.0 + (a.speed_factor < 1.0) for a in actions)
                      * c.edge_s / w.effort_ref_s)
            # Time lost: along-route progress short of sailing the nominal
            # velocity for the whole horizon, in seconds at the nominal speed.
            end = own_tr[-1]
            progress = ((end.north - states[self.own_id].north) * math.cos(psi0)
                        + (end.east - states[self.own_id].east) * math.sin(psi0))
            time_lost = max(0.0, H * c.plan_dt - progress / own_cfg.ref_speed)
            # Closest approach to land or shoal along the scored own track.
            shore = (float(np.min(c.hazards.distance(np.array([[s.north, s.east] for s in own_tr]))))
                     if c.hazards is not None else math.inf)
            ground = (w.w_ground * max(0.0, (w.d_shore - shore) / w.d_shore) ** 2
                      + w.w_aground * (shore < 0.0))

            terms = {
                "safety":   sum(p * s for (_, p), s in zip(hyps, safety)),
                "risk":     w.risk_weight * max(
                    s_h for (_, p), s_h in zip(hyps, safety) if p >= w.risk_floor or p == hyps[0][1]
                ),
                "colregs":  sum(p * cr for (_, p), cr in zip(hyps, colregs)),
                "deviation": w.w_dev * float(np.mean((xtrack / d_pass) ** 2)),
                "effort":   w.w_effort * effort,
                "time":     w.w_time * time_lost / w.time_ref_s,
                "ground":   ground,
            }
            plan = ScoredPlan(actions=actions, cost=sum(terms.values()),
                              min_dist=worst_d, terms=terms, min_shore=shore)
            scored.append((plan, rollouts, refs[i] if refs else None))

        scored.sort(key=lambda x: x[0].cost)
        best, best_rollouts, best_ref = scored[0]
        cont = self._continuation(t)
        kept = next((x for x in scored if x[0].actions == cont), None)
        if kept is not None and self._keep(kept[0], best):
            best, best_rollouts, best_ref = kept
        if best.actions != cont:          # a new plan: its edges count from now
            self._kept = (best.actions, t)
        return SearchResult(
            best       = best,
            ranked     = [x[0] for x in scored],
            hypotheses = hyps,
            own_traj   = best_ref if best_ref is not None else best_rollouts[0][self.own_id],
            rollouts   = best_rollouts,
            plan_dt    = c.plan_dt,
            kept       = kept[0] if kept is not None else None,
            hull_traj  = best_rollouts[0][self.own_id] if best_ref is not None else None,
        )

    def _keep(self, kept: ScoredPlan, fresh: ScoredPlan) -> bool:
        """Keep the plan in force over the cheapest fresh plan: unless it is
        less safe (CPA and terrain cost higher by more than hold_safety_tol),
        only a saving above the margin justifies switching."""
        w = self.cfg.weights
        def cpa(p: ScoredPlan) -> float:
            return p.terms["safety"] + p.terms["risk"] + p.terms["ground"]
        if cpa(kept) > cpa(fresh) + w.hold_safety_tol + 1e-9:
            return False
        return kept.cost - fresh.cost <= max(w.hold_abs_margin, w.hold_rel_margin * kept.cost)


def _states(pos: np.ndarray, vel: np.ndarray, psi0: float) -> list[VesselState]:
    """VesselStates from (n, 2) positions and velocities; the course is kept
    while (nearly) stopped."""
    out, psi = [], psi0
    for (n, e), (vn, ve) in zip(pos, vel):
        speed = math.hypot(vn, ve)
        if speed > 1e-3:
            psi = math.atan2(ve, vn)
        out.append(VesselState(float(n), float(e), psi, speed))
    return out


def interpolate_track(traj: list[VesselState], dt: float, t: float) -> tuple[float, float, float]:
    """(north, east, course) along a trajectory sampled every dt, at time t
    (held at the last sample beyond its end)."""
    if t <= 0.0:
        s = traj[0]
        return s.north, s.east, s.psi
    k = t / dt
    i = int(k)
    if i >= len(traj) - 1:
        s = traj[-1]
        return s.north, s.east, s.psi
    a, b, f = traj[i], traj[i + 1], k - i
    return (a.north + f * (b.north - a.north),
            a.east + f * (b.east - a.east),
            wrap_angle(a.psi + f * wrap_angle(b.psi - a.psi)))
