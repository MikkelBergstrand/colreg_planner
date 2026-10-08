"""planner_api/service.py

PredictionService: a query interface over VesselPredictor for downstream
planners.

The service runs the full predictor cycle once per observation
(update -> predict_nominal -> predict) and caches the result as a Snapshot.
All get_* methods are cheap reads of that cached snapshot and return plain
dicts/lists, so they can be called many times per step (for example by an
LLM planner through tools.call_tool) without re-running the prediction or
disturbing the responding filters.

Typical usage
-------------
    service = PredictionService.from_config("scenarios/real/window_079.yaml",
                                            dt=2.0, ref_lat=59.4, ref_lon=10.6)
    for t, obs in stream:                        # obs: {vessel_id: VesselState | dict}
        service.step(obs, t=t)
        service.get_encounters("ego")
        service.get_responsiveness()
        service.get_predicted_cpa("ego")
        service.get_predicted_trajectory("257123000", resolution_s=30)
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.colregs_graph.colregs_classifier import _compute_dcpa_tcpa
from src.colregs_graph.colregs_graph import ColregsGraph, apply_responsiveness
from src.planner_api.schema import (
    Encounter,
    Hypothesis,
    PredictedCPA,
    RoleAssignment,
    Snapshot,
    TrajectoryPoint,
    VesselStatus,
)
from src.prediction.parallel import resolve_workers
from src.prediction.predictor import VesselPredictor
from src.prediction.vo_rollout import VOAgentConfig
from src.vessel.state import Intent, VesselState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deg360(rad: float) -> float:
    return math.degrees(rad) % 360.0


def _enum_name(e) -> str | None:
    return None if e is None else e.name.lower()


# Encounter type seen from the other vessel's side.
_MIRROR_ENCOUNTER = {
    "head_on":            "head_on",
    "crossing_starboard": "crossing_port",
    "crossing_port":      "crossing_starboard",
    "overtaking_ahead":   "overtaking_astern",
    "overtaking_astern":  "overtaking_ahead",
}


def to_vessel_state(obs: VesselState | Mapping[str, Any]) -> VesselState:
    """Coerce an observation to a VesselState.

    Accepts a VesselState, or a mapping with ``north``, ``east``, ``speed``
    [m/s] and one of ``course_deg`` / ``heading_deg`` [deg, 0 = north,
    clockwise] / ``psi`` [rad]; optionally ``yaw_rate_deg_s`` or ``r`` [rad/s].
    """
    if isinstance(obs, VesselState):
        return obs
    if "psi" in obs:
        psi = float(obs["psi"])
    elif "course_deg" in obs:
        psi = math.radians(float(obs["course_deg"]))
    elif "heading_deg" in obs:
        psi = math.radians(float(obs["heading_deg"]))
    else:
        raise ValueError("observation needs one of 'psi', 'course_deg' or 'heading_deg'")
    if "r" in obs:
        r = float(obs["r"])
    else:
        r = math.radians(float(obs.get("yaw_rate_deg_s", 0.0)))
    return VesselState(
        north=float(obs["north"]),
        east=float(obs["east"]),
        psi=psi,
        speed=float(obs["speed"]),
        r=r,
    )


def build_predictor(
    pred_cfg: Mapping[str, Any],
    dt: float,
    *,
    ref_lat: float | None = None,
    ref_lon: float | None = None,
    root: str | Path = ".",
) -> VesselPredictor:
    """Build a VesselPredictor from a ``prediction:`` config block.

    Same keys as the real-data configs (see scenarios/README.md):
    ``H``, ``nominal.{enabled, library_path, beta, r, max_candidates,
    heading_cone_deg}`` and ``interaction.{enabled, use_responding_filter,
    top_k_hypotheses, reclassify_in_rollout, parallel_workers, agent}``.
    ``parallel_workers`` (int or "auto", default 0) gives the predictor a pool
    of worker processes for its rollouts and for evaluate_manoeuvres();
    "auto" uses CPU cores - 1. ``library_path`` is
    resolved against *root*. The nominal library needs ref_lat/ref_lon (the
    NED origin) to place queries in the library's UTM frame.
    """
    nominal_cfg     = pred_cfg.get("nominal", {}) or {}
    interaction_cfg = pred_cfg.get("interaction", {}) or {}
    agent_dict      = dict(interaction_cfg.get("agent", {}) or {})

    nominal_retriever = None
    if nominal_cfg.get("enabled", False):
        from src.nominal.library import TrackLibrary
        from src.nominal.retriever import NominalRetriever

        lib_path = Path(root) / nominal_cfg["library_path"]
        if not lib_path.exists():
            raise FileNotFoundError(f"nominal library not found: {lib_path}")
        if ref_lat is None or ref_lon is None:
            raise ValueError("the nominal library needs ref_lat and ref_lon (NED origin)")
        nominal_retriever = NominalRetriever(
            TrackLibrary.load(lib_path),
            **{k: nominal_cfg[k] for k in ("beta", "r", "max_candidates", "heading_cone_deg")
               if k in nominal_cfg},
        )

    predictor = VesselPredictor(
        dt                       = dt,
        H                        = pred_cfg.get("H", 30),
        default_agent_config     = VOAgentConfig.from_dict(agent_dict, ignore=("filter_rollout_model",)),
        use_responding_filter    = interaction_cfg.get("use_responding_filter", False),
        responding_rollout_model = agent_dict.get("filter_rollout_model"),
        default_top_k            = interaction_cfg.get("top_k_hypotheses", 1),
        reclassify_in_rollout    = bool(interaction_cfg.get("reclassify_in_rollout", True)),
        nominal_retriever        = nominal_retriever,
        interaction_enabled      = interaction_cfg.get("enabled", True),
        parallel_workers         = resolve_workers(interaction_cfg.get("parallel_workers", 0), None),
    )
    if nominal_retriever is not None:
        predictor.set_nominal_origin(ref_lat, ref_lon)
    return predictor


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class PredictionService:
    """Stateful prediction service with planner-friendly getters.

    Parameters
    ----------
    predictor : VesselPredictor
        The underlying predictor. The service owns its update/predict cycle;
        do not call update()/predict() on it directly.
    top_k : int, optional
        Number of responsiveness hypotheses to roll out per step. None ->
        the predictor's default_top_k.
    own_id : str, optional
        The own ship. Its responsiveness is fixed (it follows COLREGs) instead
        of estimated, so hypotheses only vary over the other vessels.
    """

    def __init__(
        self,
        predictor: VesselPredictor,
        *,
        top_k: int | None = None,
        own_id: str | None = None,
    ) -> None:
        self._predictor = predictor
        self._top_k     = top_k
        self._own_id    = own_id
        self._snapshot: Snapshot | None = None
        self._step      = -1
        self._manoeuvre_cache: dict[tuple, dict] = {}
        if own_id is not None:
            predictor.set_known_intent(own_id, Intent.RESPONSIVE)

    @property
    def own_id(self) -> str | None:
        return self._own_id

    @classmethod
    def from_config(
        cls,
        config: str | Path | Mapping[str, Any],
        *,
        dt: float,
        ref_lat: float | None = None,
        ref_lon: float | None = None,
        root: str | Path | None = None,
        top_k: int | None = None,
        own_id: str | None = None,
    ) -> "PredictionService":
        """Build from a YAML path or dict.

        *config* may be a whole run config (with a ``prediction:`` block) or
        the ``prediction`` block itself. Relative library paths are resolved
        against *root* (default: the current directory).
        """
        if isinstance(config, (str, Path)):
            with open(config) as f:
                config = yaml.safe_load(f)
        pred_cfg = config.get("prediction", config)
        predictor = build_predictor(
            pred_cfg, dt, ref_lat=ref_lat, ref_lon=ref_lon, root=root or ".",
        )
        return cls(predictor, top_k=top_k, own_id=own_id)

    @property
    def predictor(self) -> VesselPredictor:
        return self._predictor

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def step(
        self,
        observations: Mapping[str, VesselState | Mapping[str, Any]],
        t: float | None = None,
    ) -> Snapshot:
        """Ingest one set of observations and recompute the predictions.

        Call once per timestep at the predictor's dt cadence; the responding
        filters assume evenly spaced observations.

        Parameters
        ----------
        observations : {vessel_id: VesselState | dict}
            Every vessel currently observed (see to_vessel_state for the dict
            form). Vessels may appear and disappear between steps.
        t : float, optional
            Timestamp of the observations [s]. Defaults to step * dt.
        """
        states = {str(vid): to_vessel_state(obs) for vid, obs in observations.items()}
        p = self._predictor
        self._step += 1
        t = float(t) if t is not None else self._step * p.dt

        self._manoeuvre_cache.clear()
        p.update(states)
        nominal = p.predict_nominal()     # also primes the filter's non-responsive branch
        bundles = p.predict_hypotheses(self._top_k)

        index = p.graph_index
        by_index = {i: vid for vid, i in index.items()}
        graph = p.colregs_graph or ColregsGraph()
        resp = p.responsiveness

        vessels = []
        for vid in sorted(states):
            s = states[vid]
            roles = [
                RoleAssignment(
                    counterpart=by_index[other],
                    role=role.value,
                    encounter_type=self._encounter_type_for(graph, index[vid], other),
                )
                for other, role in graph.get_roles(index[vid])
            ]
            vessels.append(VesselStatus(
                vessel_id      = vid,
                north          = s.north,
                east           = s.east,
                course_deg     = _deg360(s.psi),
                speed          = s.speed,
                yaw_rate_deg_s = math.degrees(s.r),
                moving         = s.speed >= p.min_moving_speed,
                p_responsive   = resp.get(vid),
                roles          = roles,
            ))

        encounters = []
        for e in graph.edges:
            a, b = by_index[e.vessel_a], by_index[e.vessel_b]
            sa, sb = states[a], states[b]
            dcpa, tcpa = _compute_dcpa_tcpa(sa, sb)
            bearing = math.atan2(sb.east - sa.east, sb.north - sa.north)
            encounters.append(Encounter(
                vessel_a       = a,
                vessel_b       = b,
                encounter_type = _enum_name(e.encounter_type),
                role_a         = e.role_a.value,
                role_b         = e.role_b.value,
                range_m        = math.hypot(sb.north - sa.north, sb.east - sa.east),
                bearing_a_to_b_deg          = _deg360(bearing),
                relative_bearing_a_to_b_deg = _deg360(bearing - sa.psi),
                dcpa_m         = dcpa,
                tcpa_s         = tcpa if math.isfinite(tcpa) else None,
            ))

        hypotheses = []
        for rank, (trajs, assignment, weight) in enumerate(bundles):
            if assignment:
                intents = {index[vid]: it for vid, it in assignment.items()}
                eff_graph = apply_responsiveness(graph, intents)
            else:
                eff_graph = graph
            hypotheses.append(Hypothesis(
                rank        = rank,
                probability = weight,
                assumptions = {vid: it.name.lower() for vid, it in assignment.items()},
                effective_roles = [
                    {
                        "vessel_a": by_index[e.vessel_a],
                        "vessel_b": by_index[e.vessel_b],
                        "role_a": e.role_a.value,
                        "role_b": e.role_b.value,
                        "overridden_for_non_responsive": e.has_non_responsive,
                    }
                    for e in eff_graph.edges
                ],
                trajectories = {vid: self._to_points(tr) for vid, tr in trajs.items()},
            ))

        self._snapshot = Snapshot(
            t          = t,
            step       = self._step,
            dt         = p.dt,
            horizon_s  = p.horizon_steps * p.dt,
            vessels    = vessels,
            encounters = encounters,
            hypotheses = hypotheses,
            nominal    = {vid: self._to_points(tr) for vid, tr in nominal.items()},
        )
        return self._snapshot

    # ------------------------------------------------------------------
    # Queries (all read the cached snapshot)
    # ------------------------------------------------------------------

    @property
    def snapshot(self) -> Snapshot:
        if self._snapshot is None:
            raise RuntimeError("no observations yet: call step() first")
        return self._snapshot

    def get_snapshot(
        self,
        include_trajectories: bool = False,
        include_nominal: bool = False,
        resolution_s: float | None = None,
    ) -> dict:
        """Full picture at the current step.

        Trajectories are left out by default because they dominate the size;
        pass include_trajectories=True (optionally with resolution_s) to get them.
        """
        d = self.snapshot.to_dict(include_trajectories=False)
        if include_trajectories:
            for h, hd in zip(self.snapshot.hypotheses, d["hypotheses"]):
                hd["trajectories"] = {
                    vid: self._resample(pts, resolution_s) for vid, pts in h.trajectories.items()
                }
        if include_nominal:
            d["nominal"] = {
                vid: self._resample(pts, resolution_s)
                for vid, pts in self.snapshot.nominal.items()
            }
        return d

    def list_vessels(self) -> list[dict]:
        """Current status of every observed vessel."""
        return [v.to_dict() for v in self.snapshot.vessels]

    def get_vessel(self, vessel_id: str) -> dict:
        """Current state, p_responsive and COLREGs roles of one vessel."""
        return self._vessel(vessel_id).to_dict()

    def get_responsiveness(self, vessel_id: str | None = None) -> dict[str, float | None]:
        """P(responsive) per vessel (or for one vessel).

        Values near 1 mean the vessel is behaving as its COLREGs obligations
        predict; values near 0 mean it is holding its nominal course
        regardless of traffic. None when the responding filter is disabled.
        """
        vessels = [self._vessel(vessel_id)] if vessel_id is not None else self.snapshot.vessels
        return {v.vessel_id: v.p_responsive for v in vessels}

    def get_encounters(self, vessel_id: str | None = None) -> list[dict]:
        """Active COLREGs encounters, optionally only those involving vessel_id.

        When vessel_id is given, each encounter is oriented so that vessel_a
        is vessel_id (roles, encounter type and bearings from its perspective).
        """
        if vessel_id is None:
            return [e.to_dict() for e in self.snapshot.encounters]
        self._vessel(vessel_id)
        out = []
        for e in self.snapshot.encounters:
            if e.vessel_a == vessel_id:
                out.append(e.to_dict())
            elif e.vessel_b == vessel_id:
                out.append(self._flip(e))
        return out

    def get_hypotheses(
        self,
        include_trajectories: bool = False,
        resolution_s: float | None = None,
    ) -> list[dict]:
        """Responsiveness hypotheses (most likely first) with their effective roles."""
        out = []
        for h in self.snapshot.hypotheses:
            d = h.to_dict(include_trajectories=False)
            if include_trajectories:
                d["trajectories"] = {
                    vid: self._resample(pts, resolution_s) for vid, pts in h.trajectories.items()
                }
            out.append(d)
        return out

    def get_predicted_trajectory(
        self,
        vessel_id: str,
        hypothesis: int = 0,
        resolution_s: float | None = None,
        horizon_s: float | None = None,
    ) -> list[dict]:
        """Interaction-aware predicted trajectory of one vessel.

        hypothesis : rank of the hypothesis (0 = most likely).
        resolution_s : spacing of the returned points [s]; None -> every dt.
        horizon_s : cut the trajectory at this time [s]; None -> full horizon.
        """
        pts = self._trajectory(vessel_id, hypothesis)
        if horizon_s is not None:
            pts = [p for p in pts if p.t <= horizon_s + 1e-9]
        return self._resample(pts, resolution_s)

    def get_nominal_trajectory(
        self,
        vessel_id: str,
        resolution_s: float | None = None,
    ) -> list[dict]:
        """Where the vessel would go without other traffic (nominal layer only)."""
        if vessel_id not in self.snapshot.nominal:
            self._vessel(vessel_id)
            raise KeyError(f"no nominal trajectory for vessel '{vessel_id}'")
        return self._resample(self.snapshot.nominal[vessel_id], resolution_s)

    def get_predicted_state(self, vessel_id: str, t: float, hypothesis: int = 0) -> dict:
        """Predicted state of one vessel t seconds ahead (linearly interpolated)."""
        pts = self._trajectory(vessel_id, hypothesis)
        if not 0.0 <= t <= pts[-1].t:
            raise ValueError(f"t={t} outside the prediction horizon [0, {pts[-1].t}]")
        times = [p.t for p in pts]
        k = min(bisect.bisect_right(times, t), len(pts) - 1)
        p0, p1 = pts[max(k - 1, 0)], pts[k]
        a = 0.0 if p1.t == p0.t else (t - p0.t) / (p1.t - p0.t)
        dc = (p1.course_deg - p0.course_deg + 180.0) % 360.0 - 180.0
        return TrajectoryPoint(
            t          = t,
            north      = p0.north + a * (p1.north - p0.north),
            east       = p0.east  + a * (p1.east  - p0.east),
            course_deg = (p0.course_deg + a * dc) % 360.0,
            speed      = p0.speed + a * (p1.speed - p0.speed),
        ).to_dict()

    def get_predicted_cpa(
        self,
        vessel_id: str,
        other_id: str | None = None,
        hypothesis: int = 0,
    ) -> list[dict]:
        """Closest approach between vessel_id and every other vessel (or other_id)
        along the predicted trajectories of one hypothesis, sorted by distance.

        Unlike the encounter DCPA (straight-line extrapolation), this accounts
        for the predicted COLREGs manoeuvres of all vessels.
        """
        trajs = self._hypothesis(hypothesis).trajectories
        own = self._trajectory(vessel_id, hypothesis)
        others = [other_id] if other_id is not None else [v for v in trajs if v != vessel_id]
        out = []
        for oid in others:
            oth = self._trajectory(oid, hypothesis)
            n = min(len(own), len(oth))
            a = np.array([[p.north, p.east] for p in own[:n]])
            b = np.array([[p.north, p.east] for p in oth[:n]])
            d = np.linalg.norm(a - b, axis=1)
            k = int(np.argmin(d))
            out.append(PredictedCPA(
                vessel_id       = vessel_id,
                other_id        = oid,
                hypothesis_rank = hypothesis,
                min_distance_m  = float(d[k]),
                t_min_s         = own[k].t,
                position_vessel = (own[k].north, own[k].east),
                position_other  = (oth[k].north, oth[k].east),
            ))
        out.sort(key=lambda c: c.min_distance_m)
        return [c.to_dict() for c in out]

    def evaluate_manoeuvre(
        self,
        vessel_id: str,
        course_change_deg: float = 0.0,
        speed_factor: float = 1.0,
        duration_s: float = 180.0,
        start_s: float = 0.0,
    ) -> dict:
        """Predict the outcome if vessel_id starts a manoeuvre start_s from now.

        The vessel keeps its nominal course until start_s, alters course by
        course_change_deg (positive = starboard) and scales its speed by
        speed_factor for duration_s, then returns to its nominal course. Every other vessel reacts to it according to its
        COLREGs role, under each hypothesis in the current snapshot.

        Returns the closest approach to every other vessel per hypothesis,
        and the worst case over all hypotheses.
        """
        return self.evaluate_manoeuvres(
            vessel_id,
            [dict(course_change_deg=course_change_deg, speed_factor=speed_factor,
                  duration_s=duration_s, start_s=start_s)],
        )[0]

    def evaluate_manoeuvres(
        self,
        vessel_id: str,
        manoeuvres: Iterable[Mapping[str, float]],
    ) -> list[dict]:
        """evaluate_manoeuvre() for several manoeuvres, in the same order.

        Each manoeuvre is a mapping with any of course_change_deg,
        speed_factor, duration_s and start_s. All rollouts are submitted
        before any is collected, so they run in parallel when the predictor
        has a worker pool (``parallel_workers``).
        """
        from src.planner_api.manoeuvres import Manoeuvre, velocity_plan

        v = self._vessel(vessel_id)
        if not v.moving:
            raise ValueError(f"vessel '{vessel_id}' is not moving")
        p = self._predictor
        hyps = self.snapshot.hypotheses
        nominal = None

        # Submit every rollout not in the cache, then collect.
        jobs = []
        for spec in manoeuvres:
            m = Manoeuvre("custom", float(spec.get("course_change_deg", 0.0)),
                          float(spec.get("speed_factor", 1.0)), float(spec.get("duration_s", 180.0)))
            start_s = float(spec.get("start_s", 0.0))
            key = (vessel_id, m.course_change_deg, m.speed_factor, m.duration_s, start_s)
            if key in self._manoeuvre_cache:
                jobs.append((key, m, start_s, None))
                continue
            if nominal is None:
                nominal = p.nominal_velocity_sequence(vessel_id)
            plan = velocity_plan(nominal, m, p.dt, start_s)
            handles = [
                p.submit_rollout_with_plans(
                    {vessel_id: plan}, {vid: Intent[a.upper()] for vid, a in h.assumptions.items()})
                for h in hyps
            ]
            jobs.append((key, m, start_s, handles))

        results = []
        for key, m, start_s, handles in jobs:
            if handles is None:
                results.append(self._manoeuvre_cache[key])
                continue
            per_hyp = []
            worst: dict[str, dict] = {}
            for h, handle in zip(hyps, handles):
                trajs = handle.result()
                own = np.array([[s.north, s.east] for s in trajs[vessel_id]])
                cpas = {}
                for oid, tr in trajs.items():
                    if oid == vessel_id:
                        continue
                    d = np.linalg.norm(own - np.array([[s.north, s.east] for s in tr]), axis=1)
                    k = int(np.argmin(d))
                    cpas[oid] = {"min_distance_m": round(float(d[k]), 1), "t_min_s": round(k * p.dt, 1)}
                    if oid not in worst or d[k] < worst[oid]["min_distance_m"]:
                        worst[oid] = {**cpas[oid], "hypothesis_rank": h.rank}
                per_hyp.append({
                    "hypothesis_rank": h.rank,
                    "probability": round(h.probability, 4),
                    "cpa": cpas,
                })

            result = {
                "vessel_id": vessel_id,
                "manoeuvre": {**{k: v for k, v in m.to_dict().items() if k != "name"},
                              "start_s": start_s},
                "worst_case_cpa": worst,
                "per_hypothesis": per_hyp,
            }
            self._manoeuvre_cache[key] = result
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _vessel(self, vessel_id: str) -> VesselStatus:
        for v in self.snapshot.vessels:
            if v.vessel_id == vessel_id:
                return v
        known = [v.vessel_id for v in self.snapshot.vessels]
        raise KeyError(f"unknown vessel '{vessel_id}'; observed: {known}")

    def _hypothesis(self, rank: int) -> Hypothesis:
        hyps = self.snapshot.hypotheses
        if not 0 <= rank < len(hyps):
            raise IndexError(f"hypothesis {rank} not available; have {len(hyps)}")
        return hyps[rank]

    def _trajectory(self, vessel_id: str, rank: int) -> list[TrajectoryPoint]:
        trajs = self._hypothesis(rank).trajectories
        if vessel_id not in trajs:
            v = self._vessel(vessel_id)
            if not v.moving:
                raise KeyError(f"vessel '{vessel_id}' is below the minimum moving speed "
                               "and is not predicted")
            raise KeyError(f"no predicted trajectory for vessel '{vessel_id}'")
        return trajs[vessel_id]

    def _to_points(self, traj: list[VesselState]) -> list[TrajectoryPoint]:
        dt = self._predictor.dt
        return [
            TrajectoryPoint(t=k * dt, north=s.north, east=s.east,
                            course_deg=_deg360(s.psi), speed=s.speed)
            for k, s in enumerate(traj)
        ]

    def _resample(self, pts: list[TrajectoryPoint], resolution_s: float | None) -> list[dict]:
        if not pts:
            return []
        stride = 1 if not resolution_s else max(1, round(resolution_s / self._predictor.dt))
        picked = pts[::stride]
        if picked[-1] is not pts[-1]:
            picked.append(pts[-1])
        return [p.to_dict() for p in picked]

    @staticmethod
    def _encounter_type_for(graph: ColregsGraph, own_idx: int, other_idx: int) -> str | None:
        for e in graph.edges:
            name = _enum_name(e.encounter_type)
            if name is None:
                continue
            if (e.vessel_a, e.vessel_b) == (own_idx, other_idx):
                return name
            if (e.vessel_b, e.vessel_a) == (own_idx, other_idx):
                return _MIRROR_ENCOUNTER[name]
        return None

    def _flip(self, e: Encounter) -> dict:
        """Encounter seen from vessel_b's perspective."""
        own_course = self._vessel(e.vessel_b).course_deg
        bearing = (e.bearing_a_to_b_deg + 180.0) % 360.0
        return Encounter(
            vessel_a       = e.vessel_b,
            vessel_b       = e.vessel_a,
            encounter_type = _MIRROR_ENCOUNTER.get(e.encounter_type),
            role_a         = e.role_b,
            role_b         = e.role_a,
            range_m        = e.range_m,
            bearing_a_to_b_deg          = bearing,
            relative_bearing_a_to_b_deg = (bearing - own_course) % 360.0,
            dcpa_m         = e.dcpa_m,
            tcpa_s         = e.tcpa_s,
        ).to_dict()
