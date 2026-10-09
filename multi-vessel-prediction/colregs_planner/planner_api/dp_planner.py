"""planner_api/dp_planner.py

ColregsDPPlanner: COLREGs-aware own-ship planning for an external vessel
system. Feed it the states of the own ship and the surrounding vessels; get
back the dynamic-positioning (DP) reference the own ship should follow.

It runs the same pipeline as the simulator's search_dp policy:

  * PredictionService estimates every other vessel's responsiveness (does it
    follow COLREGs?) from the observation stream, and supplies their
    predicted nominal motion.
  * A COLREGs classifier over all observed vessels gives the encounter roles.
  * ManoeuvreSearchPlanner chooses a manoeuvre for the own ship every
    replan_s, against the top responsiveness hypotheses, relative to the own
    ship's route (a straight leg: start point, course, speed).
  * The own ship's track along the chosen plan is the DP reference; each
    replan re-anchors it at the own ship's current position.

Frame and units: local NED metres (north, east), headings and courses in
degrees clockwise from north, speeds in m/s, time in seconds.

Typical usage
-------------
    planner = ColregsDPPlanner(
        own_id = "ego",
        route  = Route(north=0.0, east=0.0, course_deg=0.0, speed=1.5),
        dt     = 0.5,
        config = {"interaction": {"use_responding_filter": True, "top_k_hypotheses": 2,
                                  "agent": {"d_pass": 30.0, "tcpa_threshold": 100.0}}},
    )
    while running:                                   # every dt seconds
        obs = {"ego": {"north": ..., "east": ..., "course_deg": ..., "speed": ...},
               "257123000": {...}, ...}
        ref = planner.update(t, obs)                 # DPReference
        dp_controller.set_reference(ref.north, ref.east, ref.heading_deg)

    # A faster control loop can sample the committed plan in between:
    ref = planner.reference(t_now)

Limitations
-----------
* The route is one straight leg. For a route with several legs, call
  set_route() when the own ship moves onto the next leg.
* update() must be called at a fixed dt: the responsiveness filters assume
  evenly spaced observations.
* Vessels slower than the predictor's min_moving_speed (default 0.5 m/s),
  e.g. at anchor, are planned around as if holding position: they drift at
  0.05 m/s along their heading and do not react.
* A replan rolls out every plan under every hypothesis, which takes about
  0.1-1 s; give the planner worker processes (parallel_workers) to keep it
  short. update() blocks while it replans.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Optional

from colregs_planner.colregs_graph.colregs_classifier import ColregsClassifier
from colregs_planner.planner_api.manoeuvre_search import (
    ManoeuvreSearchPlanner, SearchConfig, SearchResult, interpolate_track,
)
from colregs_planner.planner_api.service import PredictionService, build_predictor, to_vessel_state
from colregs_planner.prediction.parallel import RolloutPool, resolve_workers
from colregs_planner.prediction.vo_rollout import VOAgentConfig
from colregs_planner.vessel.state import Intent, VesselState


@dataclass(frozen=True)
class Route:
    """The own ship's nominal route: a straight leg through (north, east) on
    course_deg at speed. Without traffic the own ship follows it."""

    north:      float
    east:       float
    course_deg: float
    speed:      float


@dataclass(frozen=True)
class DPReference:
    """Where the DP controller should hold the own ship at time t.

    north, east, heading_deg : reference pose (NED metres, degrees).
    vn, ve                   : reference velocity [m/s], e.g. for feedforward.
    plan                     : the committed manoeuvre sequence, e.g.
                               ("stbd_60", "nominal"); empty before the first plan.
    plan_t                   : time the plan was made [s] (None before).
    replanned                : True if this update() made a new plan.
    """

    t:           float
    north:       float
    east:        float
    heading_deg: float
    vn:          float
    ve:          float
    plan:        tuple[str, ...] = ()
    plan_t:      Optional[float] = None
    replanned:   bool = False

    def to_dict(self) -> dict:
        return {
            "t": self.t, "north": self.north, "east": self.east,
            "heading_deg": self.heading_deg, "vn": self.vn, "ve": self.ve,
            "plan": list(self.plan), "plan_t": self.plan_t, "replanned": self.replanned,
        }


class ColregsDPPlanner:
    """COLREGs-aware DP reference for the own ship from observed vessel states.

    Parameters
    ----------
    own_id : the own ship's id in the observations.
    route : the own ship's nominal route. Can be changed with set_route().
    dt : interval between update() calls [s].
    config : a ``prediction:`` block as for PredictionService (``H``,
        ``nominal``, ``interaction.{use_responding_filter, top_k_hypotheses,
        reclassify_in_rollout, agent}``). ``interaction.agent`` holds the VO
        parameters (d_pass, tcpa_threshold, ...) used for every vessel.
    own_agent : VO parameters overriding ``interaction.agent`` for the own
        ship only (e.g. its own d_pass).
    search : SearchConfig or a dict of its fields (depth, edge_s, horizon_s, ...).
    classifier : ColregsClassifier distances (d_u, dcpa_max, d_a) for the
        encounter roles; scale them to the vessels' size and speed.
    replan_s : interval between replans [s].
    parallel_workers : worker processes for the plan rollouts; 0 = serial,
        "auto" = one per rollout, at most CPU cores - 1.
    ref_lat, ref_lon, root : passed to build_predictor for the nominal track
        library (only when ``nominal.enabled``).
    """

    def __init__(
        self,
        own_id:           str,
        route:            Route,
        dt:               float,
        *,
        config:           Mapping[str, Any] | None = None,
        own_agent:        Mapping[str, Any] | None = None,
        search:           SearchConfig | Mapping[str, Any] | None = None,
        classifier:       Mapping[str, float] | None = None,
        replan_s:         float = 5.0,
        parallel_workers: int | str = 0,
        ref_lat:          float | None = None,
        ref_lon:          float | None = None,
        root:             str = ".",
    ) -> None:
        cfg = dict(config or {})
        interaction = dict(cfg.get("interaction", {}) or {})
        agent = dict(interaction.get("agent", {}) or {})
        self._agent = VOAgentConfig.from_dict(agent, ignore=("filter_rollout_model",))
        self._own_agent = VOAgentConfig.from_dict({**agent, **dict(own_agent or {})},
                                                  ignore=("filter_rollout_model",))
        self._reclassify = bool(interaction.get("reclassify_in_rollout", True))
        self._service = PredictionService(
            build_predictor(cfg, dt, ref_lat=ref_lat, ref_lon=ref_lon, root=root),
            own_id=own_id,
        )
        self._use_filter = bool(interaction.get("use_responding_filter", False))

        search_cfg = search if isinstance(search, SearchConfig) else SearchConfig(**dict(search or {}))
        n_workers = resolve_workers(
            parallel_workers, max_parallel=len(search_cfg.actions) ** search_cfg.depth * search_cfg.top_k)
        self._pool = RolloutPool(n_workers) if n_workers > 0 else None
        self._planner = ManoeuvreSearchPlanner(own_id, {}, {}, config=search_cfg, pool=self._pool)

        self._classifier = ColregsClassifier(**dict(classifier or {}))
        self._index: dict[str, int] = {own_id: 0}     # stable graph index per vessel id
        self.own_id = own_id
        self.dt = dt
        self.replan_s = replan_s
        self._route = route
        self._result: Optional[SearchResult] = None
        self._plan_t: Optional[float] = None
        self._own: Optional[VesselState] = None
        self._t: Optional[float] = None

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    @property
    def route(self) -> Route:
        return self._route

    def set_route(self, route: Route, replan: bool = True) -> None:
        """Change the own ship's nominal route (e.g. onto the next leg). With
        replan, the next update() replans at once."""
        self._route = route
        if replan:
            self._plan_t = None

    def update(
        self,
        t: float,
        observations: Mapping[str, VesselState | Mapping[str, Any]],
        force_replan: bool = False,
    ) -> DPReference:
        """Ingest the vessel states at time t and return the DP reference.

        observations : {vessel_id: VesselState or dict} for every vessel
            currently observed, including the own ship (own_id). A dict has
            ``north``, ``east``, ``speed`` and one of ``course_deg`` /
            ``heading_deg`` / ``psi`` (see to_vessel_state).
        force_replan : replan now even if replan_s has not passed, e.g. when
            a new vessel appears.

        Replans every replan_s (and on the first call); in between, the
        reference follows the committed plan.
        """
        states = {str(vid): to_vessel_state(obs) for vid, obs in observations.items()}
        if self.own_id not in states:
            raise KeyError(f"own ship '{self.own_id}' missing from the observations")
        self._service.step(states, t=t)
        for vid in states:
            self._index.setdefault(vid, len(self._index))
        graph = self._classifier.update({self._index[vid]: s for vid, s in states.items()})
        self._own, self._t = states[self.own_id], t

        replanned = False
        if force_replan or self._plan_t is None or t - self._plan_t >= self.replan_s - 1e-9:
            self._replan(t, states, graph)
            replanned = True
        return replace(self.reference(t), replanned=replanned)

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def reference(self, t: float) -> DPReference:
        """DP reference at time t along the committed plan. Before the first
        plan, the own ship's last observed state; past the plan's horizon,
        the last planned velocity is extrapolated."""
        if self._own is None:
            raise RuntimeError("call update() first")
        res = self._result
        if res is None:
            own = self._own
            tau = t - (self._t or t)
            return DPReference(t, own.north + own.vn * tau, own.east + own.ve * tau,
                               math.degrees(own.psi) % 360.0, own.vn, own.ve)
        traj, pdt = res.own_traj, res.plan_dt
        tau = max(0.0, t - self._plan_t)
        plan = tuple(a.name for a in res.best.actions)
        end = (len(traj) - 1) * pdt
        if tau >= end:
            last = traj[-1]
            extra = tau - end
            return DPReference(t, last.north + last.vn * extra, last.east + last.ve * extra,
                               math.degrees(last.psi) % 360.0, last.vn, last.ve, plan, self._plan_t)
        n, e, psi = interpolate_track(traj, pdt, tau)
        n1, e1, _ = interpolate_track(traj, pdt, min(tau + pdt, end))
        h = min(tau + pdt, end) - tau
        vn, ve = ((n1 - n) / h, (e1 - e) / h) if h > 1e-9 else (0.0, 0.0)
        return DPReference(t, n, e, math.degrees(psi) % 360.0, vn, ve, plan, self._plan_t)

    @property
    def last_plan(self) -> Optional[SearchResult]:
        """The latest SearchResult (ranked plans, cost terms, rollouts)."""
        return self._result

    @property
    def service(self) -> PredictionService:
        """The underlying PredictionService: predictions, encounters and
        responsiveness for the latest update()."""
        return self._service

    def close(self) -> None:
        """Stop the worker processes, if any."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def __enter__(self) -> "ColregsDPPlanner":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _own_config(self) -> VOAgentConfig:
        r = self._route
        return replace(self._own_agent, nominal_psi=math.radians(r.course_deg), ref_speed=r.speed,
                       origin_north=r.north, origin_east=r.east, pref_vel_seq=None)

    def _replan(self, t: float, states: dict[str, VesselState], graph) -> None:
        # Own ship: its route. Other moving vessels: the predictor's rollout
        # configs (nominal route or library velocities). Vessels too slow for
        # the predictor: holding position, non-responsive.
        predicted = self._service.predictor.rollout_configs()
        configs: dict[str, VOAgentConfig] = {self.own_id: self._own_config()}
        stationary: set[str] = set()
        for vid, s in states.items():
            if vid == self.own_id:
                continue
            if vid in predicted:
                configs[vid] = predicted[vid]
            else:
                configs[vid] = replace(self._agent, nominal_psi=s.psi, ref_speed=0.05,
                                       origin_north=s.north, origin_east=s.east, pref_vel_seq=None)
                stationary.add(vid)
        probs, intents = None, {vid: Intent.NON_RESPONSIVE for vid in stationary}
        if self._use_filter:
            resp = self._service.get_responsiveness()
            probs = {vid: 0.0 if vid in stationary else (resp.get(vid) if resp.get(vid) is not None else 1.0)
                     for vid in states}
        self._planner.configs = configs
        self._planner.index_map = {vid: self._index[vid] for vid in states}
        self._result = self._planner.plan(
            states, graph, t, responding_probs=probs, intents=intents,
            classifier=self._classifier if self._reclassify else None,
        )
        self._plan_t = t
