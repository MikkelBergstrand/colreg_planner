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

    safety_h   = sum_j  w_safe * (max(0, d_pass - d_j) / d_pass)^2
                       + w_coll * [d_j < d_coll]
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

The worst case only ranges over hypotheses with weight >= risk_floor. Over
all of them, a 1 % "the give-way vessel never acts" hypothesis would carry a
collision cost that outweighs every COLREGs term, and a stand-on own ship
would manoeuvre from the first step instead of holding course until the
filter actually doubts the other vessel (Rule 17).

Roles only count toward vessels still approaching (TCPA > 0): the classifier
keeps an encounter's roles until the vessels have opened to d_u, and once
they are past each other the rules no longer bind the return to track.
    cost = sum_h w_h (safety_h + colregs_h) + risk_weight * max_{h: w_h >= risk_floor} safety_h
         + w_dev * mean_t (cross-track offset(t) / d_pass)^2
         + w_effort * sum_edges (|offset| / 60 deg + [slowed or stopped])
         + w_time * (time lost along the route over the horizon) / edge_s
         + w_switch * [first edge differs from the action under way, which
                       started less than edge_s ago]

where d_j is the closest approach to vessel j over the horizon. The
cross-track term is averaged over the horizon rather than taken at its end:
an end-only term ties "return now" with "return on the next edge", and
replanning then postpones the return indefinitely.
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from src.colregs_graph.colregs_classifier import ColregsClassifier, _compute_dcpa_tcpa
from src.colregs_graph.colregs_graph import ColregsGraph, VesselRole, apply_responsiveness
from src.prediction.parallel import RolloutPool, submit_rollout
from src.prediction.vo_rollout import VOAgentConfig, rank_hypotheses
from src.vessel.state import Intent, VesselState, wrap_angle


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
    w_coll:      float = 1000.0
    w_port:      float = 20.0
    w_give_way:  float = 5.0
    w_stand_on:  float = 10.0
    w_dev:       float = 2.0
    w_effort:    float = 1.0
    w_switch:    float = 2.0
    w_time:      float = 1.0
    risk_weight: float = 1.0
    risk_floor:  float = 0.1    # hypotheses below this weight stay out of the worst case


@dataclass
class SearchConfig:
    depth:     int   = 2
    edge_s:    float = 15.0     # duration of each edge [s]
    plan_dt:   float = 1.0      # rollout timestep [s]
    horizon_s: float = 90.0     # rollout horizon [s]
    top_k:     int   = 2        # responsiveness hypotheses scored
    actions:   tuple[SearchAction, ...] = DEFAULT_ACTIONS
    weights:   SearchWeights = field(default_factory=SearchWeights)


@dataclass
class ScoredPlan:
    actions:   tuple[SearchAction, ...]
    cost:      float
    min_dist:  dict[str, float]          # worst case over hypotheses, per other vessel (see d_j above)
    terms:     dict[str, float]


@dataclass
class SearchResult:
    best:          ScoredPlan
    ranked:        list[ScoredPlan]       # all plans, cheapest first
    hypotheses:    list[tuple[dict[str, Intent], float]]
    own_traj:      list[VesselState]      # own ship along the best plan, plan_dt apart
    rollouts:      list[dict[str, list[VesselState]]]   # best plan, one per hypothesis
    plan_dt:       float


class ManoeuvreSearchPlanner:
    """Exhaustive search over manoeuvre sequences for one own ship.

    Parameters
    ----------
    own_id : the planning vessel.
    agent_configs : VOAgentConfig per vessel, as used by the predictor. The
        own ship's nominal_psi, ref_speed and origin define its route.
    vessel_index_map : vessel_id -> COLREGs graph index.
    own_intent : the own ship's intent. A non-responsive own ship never
        manoeuvres, so it always gets the nominal plan.
    pool : optional RolloutPool (prediction/parallel.py). Given, every plan's
        rollouts are submitted to it before any is scored, so they run in
        parallel. The result is the same either way.
    """

    def __init__(
        self,
        own_id:           str,
        agent_configs:    dict[str, VOAgentConfig],
        vessel_index_map: dict[str, int],
        own_intent:       Intent = Intent.RESPONSIVE,
        config:           SearchConfig | None = None,
        pool:             RolloutPool | None = None,
    ) -> None:
        self.own_id    = own_id
        self.configs   = agent_configs
        self.index_map = vessel_index_map
        self.own_intent = own_intent
        self.cfg       = config or SearchConfig()
        self.pool      = pool
        # Action under way and when it started: a replan may keep it going,
        # but only its first edge_s is a commitment.
        self._active: Optional[tuple[SearchAction, float]] = None

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

    def _sequences(self) -> list[tuple[SearchAction, ...]]:
        nominal = next(a for a in self.cfg.actions if a.is_nominal)
        if self.own_intent == Intent.NON_RESPONSIVE:
            return [(nominal,) * self.cfg.depth]
        return list(itertools.product(self.cfg.actions, repeat=self.cfg.depth))

    def _committed_action(self, t: float) -> Optional[SearchAction]:
        """The action under way if it started less than edge_s before t."""
        if self._active is None:
            return None
        action, started = self._active
        return action if t - started < self.cfg.edge_s - 1e-9 else None

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
            is the single hypothesis.
        classifier : if given, each rollout re-classifies on a private copy.
        """
        c, w = self.cfg, self.cfg.weights
        H = int(round(c.horizon_s / c.plan_dt))
        ids = list(states)

        if responding_probs is not None:
            probs = dict(responding_probs)
            probs[self.own_id] = 1.0 if self.own_intent == Intent.RESPONSIVE else 0.0
            hyps = rank_hypotheses(ids, probs, c.top_k)
        else:
            base = intents or {}
            hyps = [({vid: base.get(vid, Intent.RESPONSIVE) for vid in ids}, 1.0)]
        total = sum(p for _, p in hyps)
        hyps = [(a, p / total) for a, p in hyps]

        own_cfg = self.configs[self.own_id]
        d_pass, d_coll = own_cfg.d_pass, own_cfg.resolved_d_coll()
        own_idx = self.index_map[self.own_id]
        idx_to_vid = {i: vid for vid, i in self.index_map.items()}

        # Ego's roles per hypothesis, from the graph at t.
        graphs = []
        for assignment, _ in hyps:
            int_intents = {self.index_map[vid]: assignment[vid] for vid in ids}
            graphs.append((int_intents, apply_responsiveness(nom_graph, int_intents)))

        committed = self._committed_action(t)
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
            if r != VesselRole.OVERTAKING or vid not in approaching:
                continue
            dcpa, tcpa = _compute_dcpa_tcpa(on_route, states[vid])
            if 0.0 < tcpa < own_cfg.tcpa_threshold:
                overtaken.add(vid)
                overtake_conflict |= dcpa < d_overtake
                tgt = states[vid]
                lateral = (math.cos(tgt.psi) * (own_now.east - tgt.east)
                           - math.sin(tgt.psi) * (own_now.north - tgt.north))
                overtake_side[vid] = 0 if abs(lateral) < 1.0 else (1 if lateral > 0 else -1)
            else:
                overtaken_far.add(vid)
        scored: list[tuple[ScoredPlan, list[dict[str, list[VesselState]]]]] = []

        # Submit every rollout first, so that with a pool they all run in parallel.
        sequences = self._sequences()
        pending = []
        for actions in sequences:
            configs = dict(self.configs)
            configs[self.own_id] = replace(own_cfg, pref_vel_seq=self.velocity_sequence(actions))
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
                )
                for int_intents, graph in graphs
            ])

        for actions, handles in zip(sequences, pending):
            safety, colregs, worst_d = [], [], {vid: math.inf for vid in others}
            rollouts = []
            for (assignment, _), (int_intents, graph), handle in zip(hyps, graphs, handles):
                trajs = handle.result()
                rollouts.append(trajs)
                own = np.array([[s.north, s.east] for s in trajs[self.own_id]])
                s_h = 0.0
                for vid in others:
                    d = float(np.min(np.linalg.norm(
                        own - np.array([[s.north, s.east] for s in trajs[vid]]), axis=1)))
                    if vid in overtaken:
                        d_ext, t_ext = _compute_dcpa_tcpa(trajs[self.own_id][-1], trajs[vid][-1])
                        if t_ext > 0:
                            d = min(d, d_ext)
                    worst_d[vid] = min(worst_d[vid], d)
                    if vid not in overtaken_far:
                        s_h += w.w_safe * (max(0.0, d_pass - d) / d_pass) ** 2
                    s_h += w.w_coll * (d < d_coll)
                safety.append(s_h)

                c_h = 0.0
                roles = {idx_to_vid[j]: r for j, r in graph.get_roles(own_idx)
                         if idx_to_vid[j] in approaching}
                head_on = VesselRole.HEAD_ON in roles.values()
                give_way = VesselRole.GIVE_WAY in roles.values()
                give_way_open = any(r == VesselRole.GIVE_WAY and vid in give_way_conflict
                                    for vid, r in roles.items())
                head_on_open = any(r == VesselRole.HEAD_ON and vid in head_on_conflict
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
            effort = sum(abs(a.course_offset) / 60.0 + (a.speed_factor < 1.0) for a in actions)
            # Time lost: along-route progress short of sailing the nominal
            # velocity for the whole horizon, in seconds at the nominal speed.
            end = own_tr[-1]
            progress = ((end.north - states[self.own_id].north) * math.cos(psi0)
                        + (end.east - states[self.own_id].east) * math.sin(psi0))
            time_lost = max(0.0, H * c.plan_dt - progress / own_cfg.ref_speed)
            switch = committed is not None and actions[0] != committed

            terms = {
                "safety":   sum(p * s for (_, p), s in zip(hyps, safety)),
                "risk":     w.risk_weight * max(
                    s_h for (_, p), s_h in zip(hyps, safety) if p >= w.risk_floor or p == hyps[0][1]
                ),
                "colregs":  sum(p * cr for (_, p), cr in zip(hyps, colregs)),
                "deviation": w.w_dev * float(np.mean((xtrack / d_pass) ** 2)),
                "effort":   w.w_effort * effort,
                "time":     w.w_time * time_lost / c.edge_s,
                "switch":   w.w_switch * switch,
            }
            plan = ScoredPlan(actions=actions, cost=sum(terms.values()),
                              min_dist=worst_d, terms=terms)
            scored.append((plan, rollouts))

        scored.sort(key=lambda x: x[0].cost)
        best, best_rollouts = scored[0]
        if self._active is None or best.actions[0] != self._active[0]:
            self._active = (best.actions[0], t)
        return SearchResult(
            best       = best,
            ranked     = [p for p, _ in scored],
            hypotheses = hyps,
            own_traj   = best_rollouts[0][self.own_id],
            rollouts   = best_rollouts,
            plan_dt    = c.plan_dt,
        )


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
