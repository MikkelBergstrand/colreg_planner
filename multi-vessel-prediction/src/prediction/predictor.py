"""prediction/predictor.py

Stateful multi-vessel trajectory predictor for real-data use.

Feed state estimates (AIS, RADAR, …) via update() at each timestep; query
predicted trajectories via predict().  Vessels may appear and disappear
between calls — the predictor maintains a stable index registry and persists
responding filter beliefs across brief disappearances.

Typical usage
-------------
    predictor = VesselPredictor(dt=2.0, H=300, use_responding_filter=True,
                                default_agent_config=VOAgentConfig(d_pass=100.0))

    for states in observation_stream:           # dict[str, VesselState]
        p_responsive = predictor.update(states)
        bundles     = predictor.predict()       # list[(trajectories, weight)]
        best_trajs  = bundles[0][0]             # dict[str, list[VesselState]]

With parallel_workers > 0 the full-horizon hypothesis rollouts run in a pool
of worker processes (prediction/parallel.py); start_predict() returns while
they run, so a caller can go on to the next frame and collect later.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import math
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.nominal.retriever import NominalResult, NominalRetriever

import numpy as np

from src.colregs_graph.colregs_classifier import ColregsClassifier
from src.colregs_graph.colregs_graph import ColregsGraph, VesselRole, apply_responsiveness
from src.prediction.parallel import Pending, RolloutPool, submit_rollout
from src.prediction.responding_filter import RespondingFilter
from src.prediction.vo_rollout import (
    VOAgentConfig,
    enumerate_and_rollout,
    make_vo_policy,
    nominal_1step,
    start_enumerate_and_rollout,
    vo_rollout,
)
from src.vessel.state import Intent, VesselState


class VesselPredictor:
    """Stateful multi-vessel trajectory predictor.

    Parameters
    ----------
    dt : float
        Observation/prediction timestep [s].
    H : int
        Number of steps in each predicted trajectory.
    agent_configs : dict[str, VOAgentConfig], optional
        Per-vessel VO configuration.  Vessels not listed here use
        *default_agent_config* (or a bare VOAgentConfig() if None).
    default_agent_config : VOAgentConfig, optional
        Fallback config for unknown contacts.
    use_responding_filter : bool
        Whether to run the RespondingFilter and enumerate 2^M hypotheses.
    responding_kwargs : dict[str, dict], optional
        Per-vessel keyword overrides forwarded to RespondingFilter.__init__.
    default_top_k : int
        Default number of trajectory bundles returned by predict().
    known_intents : dict[str, Intent], optional
        Vessels whose responsiveness is known rather than estimated, e.g. the
        own ship. They get no responding filter, a fixed p_responsive of 1 or
        0, and are left out of the hypothesis enumeration.
    parallel_workers : int
        Worker processes for the full-horizon rollouts; 0 = run them in this
        process.  Results are identical either way.
    """

    def __init__(
        self,
        dt:                   float,
        H:                    int,
        *,
        agent_configs:        dict[str, VOAgentConfig] | None = None,
        default_agent_config: VOAgentConfig | None            = None,
        use_responding_filter:                bool                            = False,
        responding_kwargs:             dict[str, dict] | None          = None,
        responding_rollout_model:      str | None                      = None,
        default_top_k:                 int                             = 1,
        min_moving_speed:              float                           = 0.5,
        nominal_retriever:             "NominalRetriever | None"       = None,
        interaction_enabled:           bool                            = True,
        reclassify_in_rollout:         bool                            = True,
        known_intents:                 dict[str, Intent] | None        = None,
        parallel_workers:              int                             = 0,
    ) -> None:
        self._dt               = dt
        self._H                = H
        self._use_responding_filter   = use_responding_filter
        self._responding_kwargs = responding_kwargs or {}
        # Motion model for the responding filter's internal 1-step scoring
        # rollout.  None → reuse each vessel's agent-config rollout_model (the
        # same model as the full-horizon prediction).  Set it to decouple the
        # two, e.g. "cv" for the filter while the visualisation uses
        # "ct_decoupled_ar".
        self._responding_rollout_model       = responding_rollout_model
        self._default_top_k    = default_top_k
        # Re-run the COLREGs classifier at every step of the full-horizon
        # hypothesis rollouts (paper Sec. V-D).  False → hold each hypothesis
        # graph fixed for the whole horizon (faster).
        self._reclassify_in_rollout = reclassify_in_rollout
        self._min_moving_speed = min_moving_speed

        self._agent_configs        : dict[str, VOAgentConfig]     = dict(agent_configs or {})
        self._default_agent_config : VOAgentConfig | None         = default_agent_config

        self._stable_index : dict[str, int] = {}
        self._next_index   : int            = 0
        self._active_ids   : frozenset[str] = frozenset()

        self._responding_filters     : dict[str, RespondingFilter]     = {}
        self._known_intents          : dict[str, Intent]               = dict(known_intents or {})

        self._classifier         : ColregsClassifier    = ColregsClassifier()
        self._current_states     : dict[str, VesselState] = {}
        self._current_nom_graph  : ColregsGraph | None  = None

        self._nominal_retriever  : "NominalRetriever | None"   = nominal_retriever
        self._interaction_enabled: bool                        = interaction_enabled
        self._pos_history        : dict[str, deque]            = {}
        self._pos_history_step   : dict[str, int]              = {}
        self._nominal_cache      : dict[str, "NominalResult"]  = {}
        self._last_nominal_trajs : dict[str, list[VesselState]] = {}

        self._timing : dict[str, list[float]] | None = None

        self._pool: RolloutPool | None = (
            RolloutPool(parallel_workers) if parallel_workers > 0 else None
        )

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def enable_profiling(self) -> None:
        self._timing = defaultdict(list)

    def timing_report(self) -> dict[str, list[float]]:
        return dict(self._timing) if self._timing is not None else {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_vessel(self, vid: str, state: VesselState) -> None:
        if vid not in self._stable_index:
            self._stable_index[vid] = self._next_index
            self._next_index += 1

        if vid not in self._agent_configs:
            base = copy.deepcopy(self._default_agent_config or VOAgentConfig())
            if base.ref_speed is None:
                base.ref_speed = max(state.speed, 0.1)
            if base.nominal_psi is None:
                base.nominal_psi = state.psi
            if base.origin_north is None:
                base.origin_north = state.north
            if base.origin_east is None:
                base.origin_east = state.east
            self._agent_configs[vid] = base

        if (self._use_responding_filter and vid not in self._responding_filters
                and vid not in self._known_intents):
            kw: dict = {}
            kw.update(self._responding_kwargs.get(vid, {}))
            self._responding_filters[vid] = RespondingFilter(**kw)

    def _nominal_integrate(self) -> list[tuple[dict[str, list[VesselState]], float]]:
        """Forward-integrate per vessel using the configured nominal method; no VO interaction."""
        lib_dt   = self._nominal_retriever.library.config.dt if self._nominal_retriever else self._dt
        k_stride = max(1, round(lib_dt / self._dt))
        trajs: dict[str, list[VesselState]] = {}

        for vid, state in self._current_states.items():
            traj: list[VesselState] = [state]
            north, east = state.north, state.east

            nom = self._nominal_cache.get(vid) if self._nominal_retriever else None
            if nom is not None and nom.n_hist > 0:
                # Nearest-neighbour v_pref sequence.
                v_seq = np.repeat(nom.v_pref, k_stride, axis=0)[: self._H]
                for h in range(self._H):
                    vn  = float(v_seq[h, 0])
                    ve  = float(v_seq[h, 1])
                    spd = float(math.hypot(vn, ve))
                    psi = float(math.atan2(ve, vn)) if spd > 0.01 else traj[-1].psi
                    north += vn * self._dt
                    east  += ve * self._dt
                    traj.append(VesselState(north=north, east=east, psi=psi, speed=spd, r=0.0))
            else:
                # No library: LOS guidance back to the original mission track
                # line (origin frozen to the mission start, see _ensure_vessel).
                cfg = self._agent_configs[vid]
                nom_pol = make_vo_policy(replace(
                    cfg,
                    nominal_psi = cfg.nominal_psi if cfg.nominal_psi is not None else state.psi,
                    ref_speed   = cfg.ref_speed   if cfg.ref_speed   is not None else state.speed,
                ))
                cur = state
                for _ in range(self._H):
                    act   = nom_pol.select_action(cur, [])
                    north += act.speed * math.cos(act.heading) * self._dt
                    east  += act.speed * math.sin(act.heading) * self._dt
                    cur    = VesselState(north=north, east=east,
                                         psi=act.heading, speed=act.speed, r=0.0)
                    traj.append(cur)

            trajs[vid] = traj

        return [(trajs, 1.0)]

    def _moving_states(self) -> dict[str, VesselState]:
        return {
            vid: s for vid, s in self._current_states.items()
            if s.speed >= self._min_moving_speed
        }

    def _active_configs(self, moving_states: dict[str, VesselState]) -> dict[str, VOAgentConfig]:
        """Per-vessel rollout configs for the current step."""
        # Vessels in an active encounter use the graph-preferred heading stored in cfg;
        # all others predict straight ahead at current speed.
        active_configs: dict[str, VOAgentConfig] = {}
        for vid, state in moving_states.items():
            cfg = self._agent_configs[vid]
            effective = copy.copy(cfg)
            idx = self._stable_index.get(vid)
            in_encounter = (
                idx is not None
                and self._current_nom_graph is not None
                and bool(self._current_nom_graph.get_roles(idx))
                and cfg.ref_speed is not None
            )
            if in_encounter:
                effective.ref_speed = max(state.speed, cfg.ref_speed or 0.1)
            else:
                effective.ref_speed    = max(state.speed, 0.1)
                effective.nominal_psi  = cfg.nominal_psi if cfg.nominal_psi is not None else state.psi
                effective.origin_north = state.north
                effective.origin_east  = state.east
            active_configs[vid] = effective

        if self._nominal_retriever is not None:
            lib_dt = self._nominal_retriever.library.config.dt
            k_stride = max(1, round(lib_dt / self._dt))
            for vid in moving_states:
                nom = self._nominal_cache.get(vid)
                if nom is not None and nom.n_hist > 0 and vid in active_configs:
                    active_configs[vid] = copy.copy(active_configs[vid])
                    # Upsample library-dt velocities to playback-dt cadence.
                    v_up = np.repeat(nom.v_pref, k_stride, axis=0)[: self._H]
                    active_configs[vid].pref_vel_seq = v_up

        return active_configs

    def _fallback(self) -> list[tuple[dict[str, list[VesselState]], dict[str, Intent], float]]:
        trajs = {
            vid: [self._current_states[vid]] * (self._H + 1)
            for vid in self._current_states
        }
        return [(trajs, {}, 0.0)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        states: dict[str, VesselState],
    ) -> dict[str, float]:
        """Ingest new state estimates and step the responding filters.

        Parameters
        ----------
        states : dict[str, VesselState]
            Current state estimates, keyed by vessel id.  May contain any
            subset of vessels — new vessels are registered automatically.

        Returns
        -------
        dict[str, float]
            p_responsive per active vessel.  Empty dict when use_responding_filter is False.
        """
        _t = self._timing
        _t0 = time.perf_counter() if _t is not None else 0.0

        for vid, state in states.items():
            self._ensure_vessel(vid, state)

        if self._nominal_retriever is not None:
            l = self._nominal_retriever.library.config.l
            lib_dt = self._nominal_retriever.library.config.dt
            k_stride = max(1, round(lib_dt / self._dt))
            for vid, state in states.items():
                if vid not in self._pos_history:
                    self._pos_history[vid] = deque(maxlen=l)
                    self._pos_history_step[vid] = 0
                step = self._pos_history_step[vid]
                self._pos_history_step[vid] += 1
                if step % k_stride == 0:
                    self._pos_history[vid].append(np.array([state.north, state.east]))
                if (
                    len(self._pos_history[vid]) >= 2
                    and step % k_stride == 0
                    and state.speed >= self._min_moving_speed
                ):
                    result = self._nominal_retriever.query(
                        np.array(self._pos_history[vid]),
                        np.array([state.vn, state.ve]),
                    )
                    self._nominal_cache[vid] = result

        active_ids = list(states.keys())
        self._active_ids = frozenset(active_ids)

        self._current_states = dict(states)

        if _t is not None:
            _t1 = time.perf_counter(); _t["upd.ensure"].append(_t1 - _t0); _t0 = _t1

        int_keyed = {
            self._stable_index[vid]: states[vid]
            for vid in active_ids
            if states[vid].speed >= self._min_moving_speed
        }
        self._current_nom_graph = self._classifier.update(int_keyed)

        if _t is not None:
            _t1 = time.perf_counter(); _t["upd.classifier"].append(_t1 - _t0); _t0 = _t1

        if not self._use_responding_filter:
            return {}

        for vid in active_ids:
            if vid not in self._responding_filters:
                continue
            state_i  = states[vid]
            z_obs_4d = np.array([state_i.north, state_i.east, state_i.vn, state_i.ve])
            self._responding_filters[vid].step(z_obs_4d)
            # prepare_prediction() is called from predict() after the rollout

            if _t is not None:
                _t1 = time.perf_counter(); _t["upd.responding_step"].append(_t1 - _t0); _t0 = _t1

        responsiveness = self.responsiveness
        return {vid: responsiveness[vid] for vid in active_ids if vid in responsiveness}

    def predict(
        self,
        top_k: int | None = None,
    ) -> list[tuple[dict[str, list[VesselState]], float]]:
        """Predict trajectories for all currently active vessels.

        Parameters
        ----------
        top_k : int, optional
            Maximum number of hypothesis bundles to return, sorted by weight
            descending.  Defaults to *default_top_k* set at construction.

        Returns
        -------
        list of (trajectories, weight)
            Each element is a (dict[vessel_id → list[VesselState]], float).
            Trajectories contain *H + 1* states (including t=0).
            Callers that need only the single best prediction: bundles[0][0].
        """
        return [(trajs, weight) for trajs, _, weight in self.predict_hypotheses(top_k)]

    def predict_hypotheses(
        self,
        top_k: int | None = None,
    ) -> list[tuple[dict[str, list[VesselState]], dict[str, Intent], float]]:
        """Same as predict(), but keeps each bundle's responsiveness assignment.

        Returns
        -------
        list of (trajectories, assignment, weight)
            ``assignment`` maps vessel id -> Intent assumed in that hypothesis.
            It is empty when no hypotheses were enumerated (responding filter
            off, interaction disabled, or the fallback path).
        """
        return self.start_predict_hypotheses(top_k).result()

    def start_predict(self, top_k: int | None = None) -> Pending:
        """predict(), split in two: everything that updates the predictor's
        state (classifier, responding-filter preparation) is done when this
        returns; the full-horizon rollouts may still be running on the
        worker pool.  .result() on the returned Pending waits for them and
        gives predict()'s return value.

        The rollouts work on a snapshot, so update() may be called for the
        next frame before .result().
        """
        pending = self.start_predict_hypotheses(top_k)
        return Pending(lambda: [(trajs, weight) for trajs, _, weight in pending.result()])

    def start_predict_hypotheses(self, top_k: int | None = None) -> Pending:
        """start_predict() for predict_hypotheses(): .result() gives
        (trajectories, assignment, weight) bundles.
        """
        _t = self._timing
        _t0 = time.perf_counter() if _t is not None else 0.0

        if top_k is None:
            top_k = self._default_top_k

        if self._current_nom_graph is None:
            if not self._current_states:
                return Pending.done([])
            int_keyed = {self._stable_index[vid]: s for vid, s in self._current_states.items()}
            self._current_nom_graph = self._classifier.update(int_keyed)

        if not self._interaction_enabled:
            return Pending.done([(trajs, {}, w) for trajs, w in self._nominal_integrate()])

        moving_states    = self._moving_states()
        vessel_index_map = {vid: self._stable_index[vid] for vid in moving_states}
        active_configs   = self._active_configs(moving_states)

        if _t is not None:
            _t1 = time.perf_counter(); _t["prd.config_build"].append(_t1 - _t0); _t0 = _t1

        # Fallback computed now: by .result() time the current states may
        # belong to a later frame.
        fallback = self._fallback()

        if not self._use_responding_filter:
            handle = submit_rollout(
                self._pool,
                initial_states   = moving_states,
                dt               = self._dt,
                n_steps          = self._H,
                agent_configs    = active_configs,
                colregs_graph    = self._current_nom_graph,
                vessel_index_map = vessel_index_map,
                _timings         = _t,
            )

            def collect_single():
                try:
                    trajs = handle.result()
                except ValueError:
                    return fallback
                if _t is not None:
                    _t["prd.rollout_total"].append(time.perf_counter() - _t0)
                return [(trajs, {}, 1.0)]

            return Pending(collect_single)

        probs = {vid: self._responding_filters[vid].predicted_p_responsive() for vid in moving_states
                 if vid in self._responding_filters}
        probs.update(self._known_p_responsive(moving_states))
        try:
            # Cheap 1-step rollout over all 2^N hypotheses — feeds the responding
            # filters without paying the full H-step cost.
            n_bundles  = 2 ** len(moving_states)
            step_configs = active_configs
            if self._responding_rollout_model is not None:
                step_configs = {
                    vid: replace(cfg, rollout_model=self._responding_rollout_model)
                    for vid, cfg in active_configs.items()
                }
            one_step = enumerate_and_rollout(
                initial_states   = moving_states,
                responding_probs = probs,
                nom_graph        = self._current_nom_graph,
                dt               = self._dt,
                n_steps          = 1,
                agent_configs    = step_configs,
                vessel_index_map = vessel_index_map,
                top_k            = n_bundles,
            )
            for vid, filt in self._responding_filters.items():
                p_r_vid = probs.get(vid, 1.0)
                responsive_preds: list[tuple[float, np.ndarray]] = []
                for trajs, assignment, bundle_weight in one_step:
                    if vid not in trajs or len(trajs[vid]) < 2:
                        continue
                    s1 = trajs[vid][1]
                    z1 = np.array([s1.north, s1.east, s1.vn, s1.ve])
                    if assignment.get(vid) == Intent.RESPONSIVE:
                        w = bundle_weight / max(p_r_vid, 1e-9)
                        responsive_preds.append((w, z1))
                if responsive_preds:
                    nom_traj = self._last_nominal_trajs.get(vid)
                    if nom_traj is not None and len(nom_traj) > 1:
                        s = nom_traj[1]
                        nc_z = np.array([s.north, s.east, s.vn, s.ve])
                    else:
                        nc_z = nominal_1step(moving_states[vid], active_configs.get(vid), self._dt)
                    filt.prepare_prediction(responsive_preds, nc_z)
            if _t is not None:
                _t1 = time.perf_counter(); _t["prd.pred_1step"].append(_t1 - _t0); _t0 = _t1

            # Full-horizon rollout for visualization (top_k bundles only).
            pending = start_enumerate_and_rollout(
                initial_states   = moving_states,
                responding_probs = probs,
                nom_graph        = self._current_nom_graph,
                dt               = self._dt,
                n_steps          = self._H,
                agent_configs    = active_configs,
                vessel_index_map = vessel_index_map,
                top_k            = top_k,
                colregs_classifier = self._classifier if self._reclassify_in_rollout else None,
                pool             = self._pool,
                _timings         = _t,
            )
        except ValueError:
            return Pending.done(fallback)

        def collect():
            results = pending.result()
            if _t is not None:
                _t["prd.rollout_total"].append(time.perf_counter() - _t0)
            if not results:
                return fallback
            return results

        return Pending(collect)

    # ------------------------------------------------------------------
    # Properties / configuration
    # ------------------------------------------------------------------

    def set_nominal_origin(self, ref_lat: float, ref_lon: float) -> None:
        """Forward the NED origin to the NominalRetriever so library queries
        are transformed into the correct absolute UTM frame."""
        if self._nominal_retriever is not None:
            self._nominal_retriever.set_origin(ref_lat, ref_lon)

    @property
    def latched_ids(self) -> set[str]:
        """Vessel ids currently in an active encounter."""
        if self._current_nom_graph is None:
            return set()
        return {
            vid for vid, idx in self._stable_index.items()
            if bool(self._current_nom_graph.get_roles(idx))
        }

    @property
    def burden_ids(self) -> set[str]:
        """Vessel ids currently holding a non-STAND-ON COLREGs role (GIVE_WAY,
        HEAD_ON, OVERTAKING, or AVOID_COLLISION) toward at least one
        counterpart, per the raw classifier graph (``_current_nom_graph``,
        i.e. G^nom_t — before ``apply_responsiveness()`` overrides roles for
        non-responsive counterparts). A strict subset of ``latched_ids``:
        a vessel latched into an encounter with STAND_ON only is not
        "burdened" — it isn't obligated to give way or otherwise act.
        """
        if self._current_nom_graph is None:
            return set()
        return {
            vid for vid, idx in self._stable_index.items()
            if any(role != VesselRole.STAND_ON
                   for _, role in self._current_nom_graph.get_roles(idx))
        }

    @property
    def responsiveness(self) -> dict[str, float]:
        """p_responsive for every vessel that has a responding filter or a
        known intent."""
        out = {vid: f.p_responsive for vid, f in self._responding_filters.items()}
        out.update(self._known_p_responsive(self._stable_index))
        return out

    def _known_p_responsive(self, vids) -> dict[str, float]:
        return {
            vid: 1.0 if intent == Intent.RESPONSIVE else 0.0
            for vid, intent in self._known_intents.items() if vid in vids
        }

    def set_known_intent(self, vessel_id: str, intent: Intent) -> None:
        """Fix a vessel's responsiveness (e.g. the own ship) instead of estimating it."""
        self._known_intents[vessel_id] = intent
        self._responding_filters.pop(vessel_id, None)

    def rollout_with_plans(
        self,
        plans:      dict[str, np.ndarray],
        assignment: dict[str, Intent],
        n_steps:    int | None = None,
    ) -> dict[str, list[VesselState]]:
        """Roll out the current situation with some vessels executing a fixed plan.

        Call after update() (and predict()) for the current step.

        Parameters
        ----------
        plans : {vessel_id: ndarray (n, 2)}
            Planned [vn, ve] per step for each planning vessel. The vessel
            tracks it regardless of traffic; the last row is held if the
            rollout is longer. The other vessels react to it according to
            their COLREGs roles and *assignment*.
        assignment : {vessel_id: Intent}
            Responsiveness hypothesis for the other vessels (missing ->
            responsive), e.g. one of the assignments from predict_hypotheses().
        n_steps : int, optional
            Rollout length; defaults to H.

        Returns
        -------
        dict[str, list[VesselState]]
            Trajectories of all moving vessels, n_steps + 1 states each.
        """
        moving = self._moving_states()
        missing = set(plans) - set(moving)
        if missing:
            raise ValueError(f"vessels {sorted(missing)} are not moving or not observed")
        configs = self._active_configs(moving)
        for vid, seq in plans.items():
            configs[vid] = replace(configs[vid], pref_vel_seq=np.asarray(seq, dtype=float))
        index_map = {vid: self._stable_index[vid] for vid in moving}
        intents = {
            index_map[vid]: assignment.get(vid, self._known_intents.get(vid, Intent.RESPONSIVE))
            for vid in moving
        }
        graph = self._current_nom_graph or ColregsGraph()
        return vo_rollout(
            initial_states     = moving,
            dt                 = self._dt,
            n_steps            = n_steps or self._H,
            agent_configs      = configs,
            colregs_graph      = apply_responsiveness(graph, intents),
            vessel_index_map   = index_map,
            colregs_classifier = (copy.deepcopy(self._classifier)
                                  if self._reclassify_in_rollout else None),
            intents            = intents,
            fixed_plan_ids     = set(plans),
        )

    def nominal_velocity_sequence(self, vessel_id: str) -> np.ndarray:
        """(H, 2) [vn, ve] the vessel would follow without traffic: the
        library's preferred velocities when available, else its current
        velocity held constant."""
        state = self._current_states[vessel_id]
        cfg = self._active_configs({vessel_id: state})[vessel_id]
        if cfg.pref_vel_seq is not None and len(cfg.pref_vel_seq) > 0:
            seq = np.asarray(cfg.pref_vel_seq, dtype=float)
            if len(seq) < self._H:
                seq = np.vstack([seq, np.repeat(seq[-1:], self._H - len(seq), axis=0)])
            return seq[: self._H]
        return np.tile([state.vn, state.ve], (self._H, 1))

    @property
    def colregs_graph(self) -> ColregsGraph | None:
        """Most recent ColregsGraph from the classifier."""
        return self._current_nom_graph

    @property
    def active_vessels(self) -> list[str]:
        """Vessel ids observed in the most recent update() call."""
        return list(self._active_ids)

    @property
    def dt(self) -> float:
        """Prediction timestep [s]."""
        return self._dt

    @property
    def horizon_steps(self) -> int:
        """Number of steps H in each predicted trajectory (excluding t=0)."""
        return self._H

    @property
    def min_moving_speed(self) -> float:
        """Vessels slower than this [m/s] are left out of the interaction rollout."""
        return self._min_moving_speed

    @property
    def current_states(self) -> dict[str, VesselState]:
        """States passed to the most recent update() call."""
        return dict(self._current_states)

    @property
    def graph_index(self) -> dict[str, int]:
        """Vessel id -> integer index used in colregs_graph edges."""
        return dict(self._stable_index)

    @property
    def last_nominal_trajectories(self) -> dict[str, list[VesselState]]:
        """Result of the most recent predict_nominal() call."""
        return dict(self._last_nominal_trajs)

    def predict_nominal(self) -> dict[str, list[VesselState]]:
        """Run nominal-only integration (no VO interaction).

        Returns the single best nominal trajectory per vessel, caches it in
        _last_nominal_trajs so predict() can use step-1 as the NC branch
        prediction without re-computing.  Call this before predict() each step.
        """
        bundles = self._nominal_integrate()
        self._last_nominal_trajs = bundles[0][0] if bundles else {}
        return self._last_nominal_trajs
