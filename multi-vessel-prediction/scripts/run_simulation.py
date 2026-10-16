"""scripts/run_simulation.py

Entry point: load config.yaml, build vessels (each with its own sim model
and policy), simulate forward, and visualize live.

Usage
-----
    python scripts/run_simulation.py
    python scripts/run_simulation.py --config path/to/other_config.yaml
    python scripts/run_simulation.py --save-gif           # save animation to outputs/<config_stem>.gif
    python scripts/run_simulation.py --config scenarios/simulation/multi_vessel.yaml --stop-time 40
    python scripts/run_simulation.py --record             # save live playback to outputs/videos/

Video recording (viz.video in config, or --record)
----------------------------------------------------
  viz.video.enabled:      true/false — record live playback to outputs/videos/
  viz.video.fps:          int        — frame rate of the saved video
  viz.video.start_time_s: s/null     — restrict the RECORDED video to this sim-time
  viz.video.end_time_s:   s/null       window (t = step * dt). null on either side =
                                        unrestricted on that side. Live playback itself
                                        always plays the full run regardless — this only
                                        trims which frames get grabbed into the video.

  Only applies to live playback (not the --save-gif/viz.save_path gif export,
  which renders offline via FuncAnimation instead of the live loop). Frames
  are grabbed only while playback is running (not while paused), and the
  video is finalized once end_time_s is reached, or on quit/window
  close/Ctrl-C, so an interrupted run still saves whatever played up to that
  point.

Paper figures
-------------
  All typography is Times New Roman (serif). Sizes are set in the config's
  `viz` block: `font_size` (all labels/ticks/legends/titles) and
  `vessel_label_fontsize` (the V1, V2, ... map markers).

  Controls during live playback:
    Space   : pause / resume
    Q / Esc : quit
    S       : save the current frame to outputs/frames/ (or viz.frames_dir),
              format via viz.frame_format ("pdf"/"svg"/"png", default "pdf")

  Stop-at-time: set `viz.stop_time_s` (or pass --stop-time) to auto-pause and
  save a frame the moment playback reaches that sim time (t = step * dt) —
  handy for grabbing a specific figure without hunting for the right moment
  to press S by hand.

Author: Johannes Robert Skarø
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from colregs_planner.vessel.encounter import ScenarioConfig
from colregs_planner.vessel.state import VesselInit, VesselState, Intent
from colregs_planner.vessel.base import BaseSimModel, BasePolicy, Vessel
from colregs_planner.motion_models.adapters import build_adapter
from colregs_planner.policies.constant import (
    ConstantVelocityPolicy, ConstantTurnPolicy, FixedCommandPolicy,
)
from colregs_planner.policies.vo_analytic import NominalLOSPolicy
from colregs_planner.policies.cto import CTOPolicy
from colregs_planner.policies.predictor_dp import PredictorDPPolicy
from colregs_planner.policies.search_dp import SearchDPPolicy
from colregs_planner.planner_api.manoeuvre_search import (
    COST_CATEGORIES, ManoeuvreSearchPlanner, SearchConfig, SearchWeights, cost_by_category,
    interpolate_track,
)
from colregs_planner.colregs_graph.colregs_graph import apply_responsiveness
from colregs_planner.colregs_graph.colregs_classifier import ColregsClassifier
from colregs_planner.prediction.parallel import RolloutPool, resolve_workers
from colregs_planner.prediction.responding_filter import RespondingFilter
from colregs_planner.prediction.vo_rollout import (
    vo_rollout, VOAgentConfig, enumerate_and_rollout, make_vo_policy, nominal_1step,
)
from colregs_planner.visualization.live_viz import (
    PlanCosts, PlanOverlay, VesselMeta, SimInfo, VizConfig, run_live, save_responding_plot, VESSEL_COLORS,
)
from colregs_planner.scenario_config import PredictionConfig, build_vo_agent_config, load_config


# ---------------------------------------------------------------------------
# Model / policy factories
# ---------------------------------------------------------------------------

def _other_indices(config: ScenarioConfig, vessel_idx: int) -> list[int]:
    """Graph indices of every vessel except vessel_idx.  The COLREGs graph
    itself is pushed to the policies each step by simulate_scenario."""
    return [i for i in range(len(config.vessels)) if i != vessel_idx]


_PLANNED_POLICIES = ("predictor_dp", "search_dp")


def _dp_own_intent_probs(config: ScenarioConfig, vessel_cfgs: dict[str, dict]) -> dict[str, float]:
    """p(responsive) = 1 for own ships under our control (predictor_dp,
    search_dp).

    Such a vessel follows COLREGs by construction and steers along a forecast
    made for itself, so every prediction made for it (the one it follows and
    the one drawn) pins it responsive instead of using the filter's estimate.
    """
    return {
        vi.vessel_id: 1.0
        for vi in config.vessels
        if vessel_cfgs.get(vi.vessel_id, {}).get("policy") in _PLANNED_POLICIES
    }


_SEARCH_POOLS: dict[int, RolloutPool] = {}


def _search_pool(n_workers: int) -> Optional[RolloutPool]:
    """Worker pool for search_dp planners, started once and shared by every
    search vessel and run in this process (None: plan serially)."""
    if n_workers < 1:
        return None
    if n_workers not in _SEARCH_POOLS:
        _SEARCH_POOLS[n_workers] = RolloutPool(n_workers)
    return _SEARCH_POOLS[n_workers]


def _search_config(vcfg: dict) -> SearchConfig:
    """SearchConfig from a vessel's optional `search:` block: its scalar
    fields, plus a `weights:` sub-block of SearchWeights overrides, e.g.

        search:
          edge_s: 7.5
          weights: {hold_rel_margin: 0.3, w_dev: 4.0}
    """
    raw = dict(vcfg.get("search") or {})
    weights = dict(raw.pop("weights", None) or {})
    valid = {"depth", "edge_s", "plan_dt", "horizon_s", "top_k"}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"Unknown search key(s) {sorted(unknown)}. Valid: {sorted(valid | {'weights'})}")
    valid_w = set(vars(SearchWeights()))
    unknown = set(weights) - valid_w
    if unknown:
        raise ValueError(f"Unknown search.weights key(s) {sorted(unknown)}. Valid: {sorted(valid_w)}")
    return SearchConfig(**raw, weights=SearchWeights(**weights))


def _build_policy(
    vcfg:       dict,
    vi:         VesselInit,
    config:     ScenarioConfig,
    vessel_idx: int,
    pred_cfg:   Optional[PredictionConfig] = None,
    vessel_cfgs: Optional[dict[str, dict]] = None,
) -> BasePolicy:
    """Build the simulation policy for a vessel.

    VO parameters come from build_vo_agent_config, so the defaults live in
    one place (VOAgentConfig) and are shared with the rollout.
    """
    name = vcfg.get("policy", "constant_velocity")

    # A planned vessel is the own ship under our control: it always follows
    # COLREGs. A non-responsive "ego" is a test device for the other vessels'
    # reactions, so it must use a plain policy (e.g. vo_analytic).
    if name in _PLANNED_POLICIES and vi.intent == Intent.NON_RESPONSIVE:
        raise ValueError(
            f"Vessel '{vi.vessel_id}' uses policy {name}, which plans for a vessel under our "
            f"control and so always follows COLREGs; it cannot be intent: non_responsive. "
            f"Use a plain policy such as vo_analytic for a non-responsive vessel."
        )

    # Non-responsive vessels ignore traffic entirely and LOS-follow their
    # nominal mission track — no COLREGs reasoning, no collision avoidance
    # (the STAND_ON role would otherwise still react at d_coll).
    if vi.intent == Intent.NON_RESPONSIVE and name in ("vo_kuwata", "vo_analytic", "cto"):
        return NominalLOSPolicy(make_vo_policy(build_vo_agent_config(vcfg, vi)))

    if name in ("cv", "constant_velocity"):
        return ConstantVelocityPolicy()

    if name in ("ct", "constant_turn"):
        return ConstantTurnPolicy(turn_rate=float(vcfg.get("turn_rate", 0.0)))

    if name == "fixed_command":
        return FixedCommandPolicy(
            speed=float(vcfg.get("speed", vi.state.speed)),
            turn_rate=float(vcfg.get("turn_rate", 0.0)),
        )

    if name in ("vo_analytic", "vo_kuwata"):
        return make_vo_policy(
            build_vo_agent_config(vcfg, vi, policy_type=name),
            own_vessel_idx       = vessel_idx,
            other_vessel_indices = _other_indices(config, vessel_idx),
        )

    if name == "predictor_dp":
        # Fed the predictor's next step for this vessel by simulate_scenario.
        return PredictorDPPolicy(dt=config.dt)

    if name == "search_dp":
        # Plans over manoeuvre sequences with the predictor's own models; the
        # simulation calls it to replan (see simulate_scenario).
        agent_configs = {
            v.vessel_id: build_vo_agent_config(
                (vessel_cfgs or {vi.vessel_id: vcfg}).get(v.vessel_id, {}), v,
                policy_type   = pred_cfg.policy_type if pred_cfg else "vo_analytic",
                rollout_model = pred_cfg.rollout_model if pred_cfg else "ct_decoupled_ar",
            )
            for v in config.vessels
        }
        search_cfg = _search_config(vcfg)
        # prediction.parallel_workers also sizes the planner's pool: "auto"
        # gives one worker per plan rollout, at most CPU cores - 1.
        n_workers = resolve_workers(
            pred_cfg.parallel_workers if pred_cfg else 0,
            max_parallel=len(search_cfg.actions) ** search_cfg.depth * search_cfg.top_k,
        )
        planner = ManoeuvreSearchPlanner(
            own_id           = vi.vessel_id,
            agent_configs    = agent_configs,
            vessel_index_map = {v.vessel_id: i for i, v in enumerate(config.vessels)},
            config           = search_cfg,
            pool             = _search_pool(n_workers),
        )
        return SearchDPPolicy(dt=config.dt, planner=planner,
                              replan_s=float(vcfg.get("replan_s", 5.0)))

    if name == "cto":
        cfg = build_vo_agent_config(vcfg, vi)   # borrows the shared VO parameters
        return CTOPolicy(
            nominal_psi    = cfg.nominal_psi,
            ref_speed      = cfg.ref_speed,
            tcpa_threshold = cfg.tcpa_threshold,
            d_pass         = cfg.d_pass,
            los_lookahead  = cfg.los_lookahead,
            k_psi          = cfg.k_psi,
            **{k: vcfg[k] for k in ("omega_max", "tcpa_method") if k in vcfg},
        )

    raise ValueError(
        f"Unknown policy '{name}'. "
        "Choose: cv | ct | fixed_command | vo_analytic | vo_kuwata | cto | predictor_dp | search_dp"
    )


def _plan_overlay_fn(
    plan_log: dict[str, SearchDPPolicy], dt: float,
) -> Callable[[int], dict[str, PlanOverlay]]:
    """frame -> the plan each search_dp vessel was following at that frame:
    its planned track from the latest replan, marked where each manoeuvre
    starts and where it returns to its nominal course."""
    def _overlays(frame: int) -> dict[str, PlanOverlay]:
        t = frame * dt
        out: dict[str, PlanOverlay] = {}
        for vid, pol in plan_log.items():
            rec = next((p for p in reversed(pol.plans) if p.t <= t + 1e-9), None)
            if rec is None:
                continue
            res, edge_s = rec.result, pol.planner.cfg.edge_s
            marks = []
            for i, action in enumerate(res.best.actions):
                n, e, _ = interpolate_track(res.own_traj, res.plan_dt, i * edge_s)
                marks.append((n, e, action.name if i else f"{action.name} (t={rec.t:.0f} s)"))
            n, e, _ = interpolate_track(res.own_traj, res.plan_dt,
                                        len(res.best.actions) * edge_s)
            marks.append((n, e, "nominal"))
            out[vid] = PlanOverlay(
                north=[s.north for s in res.own_traj],
                east=[s.east for s in res.own_traj],
                marks=marks,
            )
        return out

    return _overlays


def _plan_costs_fn(
    plan_log: dict[str, SearchDPPolicy], dt: float, display_names: dict[str, str],
) -> Callable[[int], Optional[PlanCosts]]:
    """frame -> every plan of the latest replan of ego (else the first
    search_dp vessel), with its cost per COST_CATEGORIES group. One
    PlanCosts object per replan, so the panel redraws only when it changes."""
    vid = "ego" if "ego" in plan_log else next(iter(plan_log))
    pol = plan_log[vid]
    weights = pol.planner.cfg.weights
    legend = {cat: [(name.removeprefix("w_"), getattr(weights, name)) for name in names]
              for cat, (_, names) in COST_CATEGORIES.items()}
    cache: dict[int, PlanCosts] = {}

    def _costs(frame: int) -> Optional[PlanCosts]:
        t = frame * dt
        rec = next((p for p in reversed(pol.plans) if p.t <= t + 1e-9), None)
        if rec is None:
            return None
        if id(rec) not in cache:
            plans = rec.result.ranked
            cache[id(rec)] = PlanCosts(
                vessel_label = display_names.get(vid, vid),
                t            = rec.t,
                # The plan in force, continued (hysteresis), is marked.
                plans        = [(" → ".join(a.name for a in p.actions)
                                 + (" (kept)" if p is rec.result.kept else ""),
                                 cost_by_category(p.terms)) for p in plans],
                chosen       = next(i for i, p in enumerate(plans) if p is rec.result.best),
                weights      = legend,
            )
        return cache[id(rec)]

    return _costs


# ---------------------------------------------------------------------------
# Prediction config
# ---------------------------------------------------------------------------

def _make_nominal_pred_fn(
    vessel_inits: list[VesselInit],
    vessel_cfgs:  dict[str, dict],
    dt:           float,
    H:            int,
) -> Callable[[dict[str, VesselState], int], dict[str, list[VesselState]]]:
    """Return a pred_fn that applies LOS guidance toward each vessel's original
    mission track line, with no other vessels (nominal trajectory)."""
    # One LOS policy per vessel, origin frozen to the mission start.
    nom_policies = {
        vi.vessel_id: make_vo_policy(build_vo_agent_config(vessel_cfgs.get(vi.vessel_id, {}), vi))
        for vi in vessel_inits
    }

    def _predict(states: dict[str, VesselState], *_) -> dict[str, list[VesselState]]:
        trajs: dict[str, list[VesselState]] = {}
        for vid, state in states.items():
            pol  = nom_policies.get(vid)
            if pol is None:
                trajs[vid] = [state] * (H + 1)
                continue
            traj  = [state]
            cur   = state
            north, east = state.north, state.east
            for _ in range(H):
                act    = pol.select_action(cur, [])
                north += act.speed * math.cos(act.heading) * dt
                east  += act.speed * math.sin(act.heading) * dt
                cur    = VesselState(north=north, east=east,
                                     psi=act.heading, speed=act.speed, r=0.0)
                traj.append(cur)
            trajs[vid] = traj
        return trajs

    return _predict


def _make_pred_fn(
    vessel_inits:         list[VesselInit],
    vessel_cfgs:          dict[str, dict],
    dt:                   float,
    H:                    int,
    config:               ScenarioConfig,
    policy_type:          str,
    rollout_model:        str,
    classifier_snapshots: list[dict]                       = None,
    responding_filters:   Optional[dict] = None,
    responding_history:   Optional[dict[str, list[float]]]      = None,
    top_k:                int                                   = 1,
    force_all_responsive: bool                                  = False,
    reclassify_in_rollout: bool                                 = True,
    fixed_probs:          Optional[dict[str, float]]            = None,
    fixed_plans:          Optional[Callable[[int], dict[str, np.ndarray]]] = None,
    pool:                 Optional[RolloutPool]                 = None,
) -> Callable[[dict[str, VesselState], int], dict[str, list[VesselState]]]:
    """Return a callable that runs a VO rollout from any snapshot of states.

    The callable is computed once per animation frame — no pre-storage.
    agent_configs are built once here (mission headings are fixed).
    classifier_snapshots[k] seeds the pred classifier with the hysteresis state
    from sim step k so the rollout starts from the correct COLREGs scenario.
    When responding_filters is provided, enumerate_and_rollout is used and the
    top-1 hypothesis trajectory is returned for visualization.

    top_k : when > 1 the callable returns a *list* of ``(traj_dict, weight)``
        tuples (one per responsiveness hypothesis, most-likely first) instead of a
        single per-vessel dict, so the visualizer can overlay several hypothesis
        bundles and label each with its posterior probability.  ``weight`` is the
        joint p(hypothesis) from the responsiveness posteriors (None on fallback,
        1.0 on the plain rollout path, which only ever yields one bundle).

    reclassify_in_rollout : re-run the prediction classifier at every step of
        each hypothesis rollout (paper Sec. V-D).  False → each hypothesis graph
        is held fixed for the whole horizon (faster).

    pool : optional RolloutPool; the top-K hypothesis rollouts then run in
        parallel worker processes (identical results).

    force_all_responsive : ignore any responding filter and roll out with every
        vessel assumed responsive (no STAND_ON override) — the "assume everyone
        responds" baseline used by the DCPA evaluation.

    fixed_probs : p(responsive) pinned for these vessel_ids instead of read from
        their filters, e.g. 1.0/0.0 for a vessel whose own intent is known.

    fixed_plans : frame -> {vessel_id: (H, 2) planned [vn, ve]}. Those vessels
        execute their plan in the rollout (vo_rollout's fixed-plan mode) and
        the others react to it, e.g. a search_dp own ship's committed plan.
    """
    return_bundles = top_k > 1
    agent_configs = {
        vi.vessel_id: build_vo_agent_config(
            vessel_cfgs.get(vi.vessel_id, {}), vi,
            policy_type=policy_type, rollout_model=rollout_model,
        )
        for vi in vessel_inits
    }
    vessel_index_map = {vi.vessel_id: i for i, vi in enumerate(vessel_inits)}

    pred_classifier = ColregsClassifier(**config.classifier)

    intents = (None if force_all_responsive
               else {i: vi.intent for i, vi in enumerate(vessel_inits)})

    def _predict(states: dict[str, VesselState], frame: int = 0) -> dict[str, list[VesselState]]:
        _fallback = {vid: [s] * (H + 1) for vid, s in states.items()}

        plans = {vid: seq for vid, seq in (fixed_plans(frame) if fixed_plans else {}).items()
                 if seq is not None and vid in states}
        configs = dict(agent_configs)
        for vid, seq in plans.items():
            configs[vid] = replace(configs[vid], pref_vel_seq=seq)

        if classifier_snapshots:
            snapshot = classifier_snapshots[min(frame, len(classifier_snapshots) - 1)]
            pred_classifier._pair_states = copy.deepcopy(snapshot)
        else:
            pred_classifier.reset()

        if responding_filters is not None and not force_all_responsive:
            int_keyed = {vessel_index_map[vid]: states[vid] for vid in states}
            nom_graph = pred_classifier.update(int_keyed)
            if responding_history is not None:
                responding_probs = {
                    vid: history[min(frame, len(history) - 1)]
                    for vid, history in responding_history.items()
                }
            else:
                responding_probs = {vid: f.p_responsive for vid, f in responding_filters.items()}
            if fixed_probs:
                responding_probs.update(fixed_probs)
            try:
                bundles = enumerate_and_rollout(
                    initial_states   = states,
                    responding_probs = responding_probs,
                    nom_graph        = nom_graph,
                    dt               = dt,
                    n_steps          = H,
                    agent_configs    = configs,
                    vessel_index_map = vessel_index_map,
                    top_k            = max(1, top_k),
                    colregs_classifier = pred_classifier if reclassify_in_rollout else None,
                    fixed_plan_ids   = set(plans) or None,
                    pool             = pool,
                )
                results = ([(b[0], b[2]) for b in bundles] if bundles
                           else [(_fallback, None)])
            except ValueError:
                results = [(_fallback, None)]
            return results if return_bundles else results[0][0]

        try:
            single = vo_rollout(
                initial_states      = states,
                dt                  = dt,
                n_steps             = H,
                agent_configs       = configs,
                vessel_index_map    = vessel_index_map,
                colregs_classifier  = pred_classifier,
                intents             = intents,
                fixed_plan_ids      = set(plans) or None,
            )
        except ValueError:
            single = _fallback
        return [(single, 1.0)] if return_bundles else single

    return _predict


# ---------------------------------------------------------------------------
# DCPA evaluation
# ---------------------------------------------------------------------------

def _traj_dcpa(
    traj_a: list[VesselState],
    traj_b: list[VesselState],
) -> tuple[float, int]:
    """Minimum separation between two same-time-base trajectories and its step index."""
    n = min(len(traj_a), len(traj_b))
    best_d, best_k = math.inf, 0
    for k in range(n):
        d = math.hypot(traj_a[k].north - traj_b[k].north,
                       traj_a[k].east  - traj_b[k].east)
        if d < best_d:
            best_d, best_k = d, k
    return best_d, best_k


def evaluate_dcpa(
    config:               ScenarioConfig,
    vessel_cfgs:          dict[str, dict],
    pred_cfg:             "PredictionConfig",
    trajectories:         dict[str, list[VesselState]],
    classifier_snapshots: list[dict],
    responding_filters:   Optional[dict],
    responding_history:   Optional[dict[str, list[float]]],
    display_names:        dict[str, str],
) -> None:
    """Compare predicted vs ground-truth DCPA at each configured sim time.

    For every listed time the true vessel states are taken as the rollout
    seed and two predictions are produced:
      * top-1    : the most-likely responsiveness hypothesis (needs a responding
                   filter; degenerates to the all-responsive rollout if none)
      * all-resp : every vessel assumed responsive (responding filter off)
    The predicted distance at closest point of approach (DCPA) for each pair is
    compared against the ground-truth DCPA over the same look-ahead window.
    """
    dt      = config.dt
    horizon = pred_cfg.dcpa_eval_horizon or pred_cfg.horizon
    n_steps = max(1, round(horizon / dt))
    n_sim   = config.n_steps
    ego_id  = config.vessels[0].vessel_id

    if pred_cfg.dcpa_eval_pairs:
        pairs = [(str(a), str(b)) for a, b in pred_cfg.dcpa_eval_pairs]
    else:
        pairs = [(ego_id, vi.vessel_id) for vi in config.vessels[1:]]

    has_filter = bool(responding_filters)
    common = dict(
        vessel_inits=config.vessels, vessel_cfgs=vessel_cfgs, dt=dt,
        H=n_steps, config=config,
        policy_type=pred_cfg.policy_type,
        rollout_model=pred_cfg.rollout_model,
        classifier_snapshots=classifier_snapshots,
        top_k=1,
        reclassify_in_rollout=pred_cfg.reclassify_in_rollout,
    )
    pred_top1 = _make_pred_fn(
        **common,
        responding_filters=responding_filters if has_filter else None,
        responding_history=responding_history if has_filter else None,
        force_all_responsive=not has_filter,
        fixed_probs=_dp_own_intent_probs(config, vessel_cfgs),
    )
    pred_allresp = _make_pred_fn(
        **common,
        responding_filters=None, responding_history=None,
        force_all_responsive=True,
    )

    def _lbl(vid: str) -> str:
        return display_names.get(vid, vid)

    rows: list[tuple] = []
    print("\n" + "=" * 82)
    print("DCPA evaluation  --  predicted vs ground-truth distance at closest approach")
    if not has_filter:
        print("  (no responding filter active - 'top-1' equals 'all-resp')")
    print("=" * 82)
    for t_eval in pred_cfg.dcpa_eval_times:
        k = round(t_eval / dt)
        if not 0 <= k <= n_sim:
            print(f"\n  t = {t_eval:.1f} s : outside simulation [0, {n_sim * dt:.1f}] s - skipped")
            continue
        states  = {vid: trajectories[vid][k] for vid in trajectories}
        win     = min(n_steps + 1, n_sim + 1 - k)          # compare-window length (states)
        gt      = {vid: trajectories[vid][k:k + win] for vid in trajectories}
        top1    = pred_top1(states, k)
        allr    = pred_allresp(states, k)

        print(f"\n  t = {t_eval:.1f} s   look-ahead {(win - 1) * dt:.1f} s / {win - 1} steps")
        print(f"    {'pair':<12}{'GT':>9}{'top-1':>10}{'err':>9}"
              f"{'all-resp':>11}{'err':>9}   t_cpa(GT/top1/all)")
        for a, b in pairs:
            if a not in gt or b not in gt:
                print(f"    {a}-{b}: unknown vessel id - skipped")
                continue
            d_gt, k_gt = _traj_dcpa(gt[a],            gt[b])
            d_t1, k_t1 = _traj_dcpa(top1[a][:win],    top1[b][:win])
            d_ar, k_ar = _traj_dcpa(allr[a][:win],    allr[b][:win])
            name = f"{_lbl(a)}-{_lbl(b)}"
            print(f"    {name:<12}{d_gt:>8.1f}m{d_t1:>9.1f}m{d_t1 - d_gt:>+8.1f}m"
                  f"{d_ar:>10.1f}m{d_ar - d_gt:>+8.1f}m   "
                  f"{k_gt * dt:.0f}/{k_t1 * dt:.0f}/{k_ar * dt:.0f} s")
            rows.append((t_eval, _lbl(a), _lbl(b), d_gt, d_t1, d_ar,
                         k_gt * dt, k_t1 * dt, k_ar * dt))
    print("=" * 82)

    if pred_cfg.dcpa_eval_out and rows:
        import csv
        out_path = Path(pred_cfg.dcpa_eval_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["t_eval_s", "vessel_a", "vessel_b",
                         "dcpa_gt_m", "dcpa_top1_m", "dcpa_all_responsive_m",
                         "t_cpa_gt_s", "t_cpa_top1_s", "t_cpa_all_responsive_s"])
            wr.writerows(rows)
        print(f"  DCPA table written to {out_path}")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_scenario(
    config:                        ScenarioConfig,
    vessel_cfgs:                   dict[str, dict],
    track_responding:              bool,
    responding_rollout_model:      str,
    pred_cfg:                      Optional[PredictionConfig] = None,
    plan_log:                      Optional[dict[str, SearchDPPolicy]] = None,
) -> tuple[dict[str, list[VesselState]], dict[str, list], list, list[dict], dict, dict[str, list[float]]]:
    """Build Vessel objects from config and simulate forward.

    Returns (trajectories, policy_debug, colregs_graphs, classifier_snapshots,
             responding_filters, responding_history).
    colregs_graphs[k] is the ColregsGraph active at timestep k (n_steps+1 entries).
    classifier_snapshots[k] is a deep copy of the COLREGs classifier's
    _pair_states at step k, used to seed the prediction classifier at the
    correct animation frame.
    responding_filters maps vessel_id -> RespondingFilter (final posterior via p_responsive).
    responding_history maps vessel_id -> list of p_responsive at each step.
    Both are empty dicts when track_responding is False.

    Vessels with policy 'predictor_dp' are driven by the predictor itself: each
    step the top-1 prediction (pred_cfg's policy and rollout model, the live
    responding filters and this step's classifier state) is rolled out one
    step, and that vessel's predicted step becomes its DP reference step.

    Vessels with policy 'search_dp' replan every replan_s: the manoeuvre search
    scores its candidate plans with the same prediction model and filters, and
    the vessel's DP reference follows the chosen plan. If plan_log is given,
    it receives {vessel_id: SearchDPPolicy}; the policies keep every plan made.
    """
    from colregs_planner.colregs_graph.colregs_graph import ColregsGraph as _CG
    rng = np.random.default_rng(config.seed)

    vessels: list[Vessel] = []
    for vessel_idx, vi in enumerate(config.vessels):
        vcfg = vessel_cfgs.get(vi.vessel_id, {})
        vessels.append(Vessel.create(
            vessel_id = vi.vessel_id,
            initial   = vi.state,
            sim_model = build_adapter(vcfg.get("sim_model", "ct_decoupled"), config.dt,
                                      k_psi=vcfg.get("k_psi", VOAgentConfig.k_psi)),
            policy    = _build_policy(vcfg, vi, config=config, vessel_idx=vessel_idx,
                                      pred_cfg=pred_cfg, vessel_cfgs=vessel_cfgs),
        ))

    classifier = ColregsClassifier(**config.classifier)

    dp_vessels = [i for i, v in enumerate(vessels) if isinstance(v.policy, PredictorDPPolicy)]
    search_vessels = [i for i, v in enumerate(vessels) if isinstance(v.policy, SearchDPPolicy)]
    for i in dp_vessels + search_vessels:
        vcfg  = vessel_cfgs.get(config.vessels[i].vessel_id, {})
        model = vcfg.get("sim_model", "ct_decoupled")
        if model != "milliampere":
            raise ValueError(
                f"Vessel '{config.vessels[i].vessel_id}' uses policy {vcfg.get('policy')}, which "
                f"needs sim_model 'milliampere' (DP control), not '{model}'."
            )
    if plan_log is not None:
        plan_log.update({config.vessels[i].vessel_id: vessels[i].policy for i in search_vessels})
    # --- Responding filter setup ------------------------------------------------
    responding_filters: dict[str, RespondingFilter]      = {}
    responding_history: dict[str, list[float]]           = {}
    pred_agent_configs:  dict[str, VOAgentConfig] | None = None
    pred_vessel_idx_map: dict[str, int] | None           = None
    if track_responding:
        pred_agent_configs  = {}
        pred_vessel_idx_map = {vi.vessel_id: i for i, vi in enumerate(config.vessels)}
        for vessel_idx, vi in enumerate(config.vessels):
            vcfg = vessel_cfgs.get(vi.vessel_id, {})
            cfg  = build_vo_agent_config(vcfg, vi, policy_type="vo_analytic",
                                          rollout_model=responding_rollout_model)
            pred_agent_configs[vi.vessel_id] = cfg
            responding_filters[vi.vessel_id] = RespondingFilter()
            responding_history[vi.vessel_id] = []
    # -------------------------------------------------------------------------

    trajectories: dict[str, list[VesselState]] = {
        v.vessel_id: [v.state] for v in vessels
    }
    policy_debug: dict[str, list] = {v.vessel_id: [] for v in vessels}

    colregs_graphs: list[_CG] = []
    classifier_snapshots: list[dict] = []

    dp_pred_fn: Optional[Callable] = None
    if dp_vessels:
        # One-step version of the predictor run_live draws (_make_pred_fn); a
        # rollout is causal, so its first step equals that of the full horizon.
        # It reads classifier_snapshots and the filters live, as they update.
        dp_pred_fn = _make_pred_fn(
            config.vessels, vessel_cfgs, config.dt, 1, config,
            policy_type   = pred_cfg.policy_type if pred_cfg else "vo_analytic",
            rollout_model = pred_cfg.rollout_model if pred_cfg else responding_rollout_model,
            classifier_snapshots = classifier_snapshots,
            responding_filters   = responding_filters or None,
            top_k                = 1,
            reclassify_in_rollout = pred_cfg.reclassify_in_rollout if pred_cfg else True,
            fixed_probs = _dp_own_intent_probs(config, vessel_cfgs),
        )

    for _ in range(config.n_steps):
        # All vessels see the same snapshot of states before any of them steps
        snapshot = [v.state for v in vessels]

        if config.noise_std_pos > 0 or config.noise_std_vel > 0:
            obs_snapshot = [
                VesselState(
                    north = s.north + rng.normal(0, config.noise_std_pos),
                    east  = s.east  + rng.normal(0, config.noise_std_pos),
                    psi   = s.psi,
                    speed = max(0.0, s.speed + rng.normal(0, config.noise_std_vel)),
                    r     = s.r,
                )
                for s in snapshot
            ]
        else:
            obs_snapshot = snapshot

        int_keyed = {i: s for i, s in enumerate(obs_snapshot)}
        nom_graph  = classifier.update(int_keyed)

        if responding_filters:
            # ---- 1-step predictive responding filter --------------------
            # Step each filter with the current observation, then run a
            # 1-step rollout over all 2^N hypotheses to extract per-vessel
            # 1-step-ahead predictions for the next step.
            str_snapshot = {vi.vessel_id: obs_snapshot[i]
                            for i, vi in enumerate(config.vessels)}

            for vessel_idx, vi in enumerate(config.vessels):
                vid      = vi.vessel_id
                state_i  = obs_snapshot[vessel_idx]
                filt     = responding_filters[vid]
                z_obs_4d = np.array([state_i.north, state_i.east, state_i.vn, state_i.ve])
                filt.step(z_obs_4d)
                responding_history[vid].append(filt.p_responsive)

            # Capture beliefs AFTER filt.step() so rollout weights reflect
            # the current posterior, not the prior from the previous step.
            probs = {vid: responding_filters[vid].p_responsive
                     for vid in str_snapshot}

            # 1-step rollout over all 2^N hypotheses → extract predictions
            n_bundles  = 2 ** len(config.vessels)
            try:
                pred_results = enumerate_and_rollout(
                    initial_states   = str_snapshot,
                    responding_probs = probs,
                    nom_graph        = nom_graph,
                    dt               = config.dt,
                    n_steps          = 1,
                    agent_configs    = pred_agent_configs,
                    vessel_index_map = pred_vessel_idx_map,
                    top_k            = n_bundles,
                )
                for vi in config.vessels:
                    vid   = vi.vessel_id
                    filt  = responding_filters[vid]
                    p_r   = probs[vid]
                    responsive_preds: list[tuple[float, np.ndarray]] = []
                    for trajs, assignment, bundle_weight in pred_results:
                        if vid not in trajs or len(trajs[vid]) < 2:
                            continue
                        if assignment.get(vid) == Intent.RESPONSIVE:
                            s1 = trajs[vid][1]
                            z1 = np.array([s1.north, s1.east, s1.vn, s1.ve])
                            w = bundle_weight / max(p_r, 1e-9)
                            responsive_preds.append((w, z1))
                    if responsive_preds:
                        nc_z = nominal_1step(str_snapshot[vid], pred_agent_configs.get(vid), config.dt)
                        filt.prepare_prediction(responsive_preds, nc_z)
            except Exception:
                pass
            # -------------------------------------------------------------

        # Apply responsiveness post-processing
        intents    = {i: vi.intent for i, vi in enumerate(config.vessels)}
        step_graph = apply_responsiveness(nom_graph, intents)
        classifier_snapshots.append(copy.deepcopy(classifier._pair_states))

        # Build estimated-responsiveness graph before setting vessel policies so
        # responsive vessels use the filter's belief (not GT) for their VO
        # decisions — making the debug VO panel consistent with the colregs panel.
        if responding_filters:
            top1_intents = {
                i: (Intent.RESPONSIVE if responding_filters[vi.vessel_id].p_responsive > 0.5
                    else Intent.NON_RESPONSIVE)
                for i, vi in enumerate(config.vessels)
            }
            viz_graph = apply_responsiveness(nom_graph, top1_intents)
        else:
            viz_graph = step_graph

        # Per-vessel graph: own intent is always GT; other vessels' intents
        # come from the filter estimate.  This ensures a responsive vessel can
        # never behave as non-responsive due to a false filter classification.
        for vessel_idx, v in enumerate(vessels):
            if hasattr(v.policy, "set_colregs_graph"):
                vi = config.vessels[vessel_idx]
                if responding_filters:
                    per_intents = dict(top1_intents)
                    per_intents[vessel_idx] = vi.intent  # own slot → GT
                    v.policy.set_colregs_graph(apply_responsiveness(nom_graph, per_intents))
                else:
                    v.policy.set_colregs_graph(step_graph)

        colregs_graphs.append(viz_graph)

        if dp_pred_fn is not None:
            pred = dp_pred_fn(
                {vi.vessel_id: obs_snapshot[i] for i, vi in enumerate(config.vessels)},
                len(classifier_snapshots) - 1,
            )
            for i in dp_vessels:
                traj = pred.get(config.vessels[i].vessel_id, [])
                if len(traj) >= 2:
                    vessels[i].policy.set_predicted_step(traj[0], traj[1])

        for i in search_vessels:
            pol = vessels[i].policy
            if pol.needs_replan():
                pol.set_plan(pol.planner.plan(
                    states = {vi.vessel_id: obs_snapshot[j] for j, vi in enumerate(config.vessels)},
                    nom_graph = nom_graph,
                    t = pol.time,
                    responding_probs = ({vid: f.p_responsive for vid, f in responding_filters.items()}
                                        if responding_filters else None),
                    intents = {vi.vessel_id: vi.intent for vi in config.vessels},
                    classifier = (classifier if pred_cfg is None or pred_cfg.reclassify_in_rollout
                                  else None),
                ))

        for i, v in enumerate(vessels):
            others = [s for j, s in enumerate(snapshot) if j != i]
            v.step(others, config.dt)
            trajectories[v.vessel_id].append(v.state)
            policy_debug[v.vessel_id].append(
                v.policy.last_debug if hasattr(v.policy, "last_debug") else None
            )

    # Duplicate last entries so length matches trajectories (n_steps + 1 frames)
    colregs_graphs.append(colregs_graphs[-1] if colregs_graphs else _CG())
    if classifier_snapshots:
        classifier_snapshots.append(classifier_snapshots[-1])

    if config.noise_std_pos > 0 or config.noise_std_vel > 0:
        for vid, traj in trajectories.items():
            trajectories[vid] = [
                VesselState(
                    north = s.north + rng.normal(0, config.noise_std_pos),
                    east  = s.east  + rng.normal(0, config.noise_std_pos),
                    psi   = s.psi,
                    speed = max(0.0, s.speed + rng.normal(0, config.noise_std_vel)),
                    r     = s.r,
                )
                for s in traj
            ]

    print("\nPolicy timing:")
    for v in vessels:
        times = getattr(v.policy, "_call_times", [])
        if times:
            avg_ms = 1000.0 * sum(times) / len(times)
            policy_name = type(v.policy).__name__
            print(f"  {v.vessel_id:12s}  {policy_name:25s}  avg={avg_ms:.3f} ms  n={len(times)}")

    if responding_filters:
        print("\nResponsiveness posteriors (end of sim):")
        for vid, f in responding_filters.items():
            print(f"  {vid:12s}  p(responsive)={f.p_responsive:.3f}")

    return trajectories, policy_debug, colregs_graphs, classifier_snapshots, responding_filters, responding_history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "scenarios/simulation/head_on.yaml",
    )
    parser.add_argument("--save-gif", "--save", dest="save_gif", action="store_true",
                        help="Save animation to outputs/<scenario_name>.gif")
    parser.add_argument("--record", action="store_true",
                        help="Record live playback to outputs/videos/ "
                             "(overrides viz.video.enabled in config)")
    parser.add_argument("--ego-policy", default=None,
                        help="Override the ego vessel's policy from the config, "
                             "e.g. predictor_dp or search_dp")
    parser.add_argument("--stop-time", type=float, default=None,
                        help="Simulation time [s] at which to auto-pause and save "
                             "the current frame (format via viz.frame_format; overrides "
                             "viz.stop_time_s in config)")
    args = parser.parse_args()

    config, vessel_cfgs, vc, pred_cfg = load_config(args.config)
    if args.ego_policy:
        vessel_cfgs.setdefault("ego", {})["policy"] = args.ego_policy
    if args.save_gif:
        out_dir = Path(__file__).resolve().parents[1] / "outputs"
        out_dir.mkdir(exist_ok=True)
        vc.save_path = str(out_dir / f"{args.config.stem}.gif")
    if args.record:
        vc.video_enabled = True

    print(f"Scenario : {config.name}")
    for vi in config.vessels:
        vcfg = vessel_cfgs.get(vi.vessel_id, {})
        print(f"  {vi.vessel_id:12s}  "
              f"model={vcfg.get('sim_model', 'ct_decoupled'):14s}  "
              f"policy={vcfg.get('policy', 'constant_velocity')}")
    print(f"Duration : {config.t_horizon} s  |  dt={config.dt} s  |  steps={config.n_steps}")

    plan_log: dict[str, SearchDPPolicy] = {}
    (trajectories, policy_debug, colregs_graphs, classifier_snapshots,
     responding_filters, responding_history) = simulate_scenario(
        config, vessel_cfgs,
        track_responding=pred_cfg.use_responding_filter,
        responding_rollout_model=(pred_cfg.filter_rollout_model or pred_cfg.rollout_model),
        pred_cfg=pred_cfg,
        plan_log=plan_log,
    )
    for vid, pol in plan_log.items():
        print(f"\nManoeuvre search ({vid}): {len(pol.plans)} plans")
        for rec in pol.plans:
            b = rec.result.best
            print(f"  t={rec.t:5.1f} s  {' -> '.join(a.name for a in b.actions):20s}"
                  f"  {'kept ' if b is rec.result.kept else 'fresh'}"
                  f"  cost {b.cost:7.2f}  min dist "
                  + ", ".join(f"{o} {d:.1f} m" for o, d in b.min_dist.items()))

    pred_fn:         Optional[Callable] = None
    nominal_pred_fn: Optional[Callable] = None
    pred_every: int                     = 1
    if pred_cfg.enabled:
        H            = max(1, round(pred_cfg.horizon   / config.dt))
        pred_every   = max(1, round(pred_cfg.update_dt / config.dt))
        print(f"VO prediction enabled  "
              f"(horizon={pred_cfg.horizon} s, {H} steps, "
              f"update every {pred_cfg.update_dt} s / {pred_every} frames)")
        # At most min(top_k, 2^N) hypothesis rollouts run at the same time.
        n_workers = resolve_workers(
            pred_cfg.parallel_workers,
            max_parallel=min(pred_cfg.top_k_hypotheses, 2 ** len(config.vessels)),
        )
        if n_workers > 0:
            print(f"  Parallel rollouts : {n_workers} worker processes")
        pred_fn = _make_pred_fn(
            config.vessels, vessel_cfgs, config.dt, H, config,
            policy_type=pred_cfg.policy_type,
            rollout_model=pred_cfg.rollout_model,
            classifier_snapshots=classifier_snapshots,
            responding_filters=responding_filters if pred_cfg.use_responding_filter else None,
            responding_history=responding_history if pred_cfg.use_responding_filter else None,
            top_k=pred_cfg.top_k_hypotheses,
            reclassify_in_rollout=pred_cfg.reclassify_in_rollout,
            fixed_probs=_dp_own_intent_probs(config, vessel_cfgs),
            # A search_dp own ship is drawn executing the plan it has committed
            # to, with the other vessels reacting to that plan.
            fixed_plans=(
                (lambda frame: {vid: pol.planned_velocities(frame * config.dt, config.dt, H)
                                for vid, pol in plan_log.items()})
                if plan_log else None
            ),
            pool=RolloutPool(n_workers) if n_workers > 0 else None,
        )
        if pred_cfg.top_k_hypotheses > 1:
            print(f"  Top-K hypotheses : drawing up to {pred_cfg.top_k_hypotheses} "
                  f"trajectory bundles live")
        if pred_cfg.show_nominal_comparison:
            nominal_pred_fn = _make_nominal_pred_fn(
                config.vessels, vessel_cfgs, config.dt, H,
            )

    debug_vessel_ids = [
        vid for vid, frames in policy_debug.items()
        if any(f is not None for f in frames)
    ]

    # Display names shown on the map/legend/graph: V1, V2, ... in config order
    # (ego = V1), matching the AIS plots. Internal dict keys (trajectories,
    # predictions, debug) stay as the config vessel_ids ("ego", "target_1", ...).
    display_names = {vi.vessel_id: f"V{i+1}" for i, vi in enumerate(config.vessels)}

    if pred_cfg.dcpa_eval_enabled and pred_cfg.dcpa_eval_times:
        evaluate_dcpa(
            config, vessel_cfgs, pred_cfg, trajectories,
            classifier_snapshots=classifier_snapshots,
            responding_filters=responding_filters if pred_cfg.use_responding_filter else None,
            responding_history=responding_history if pred_cfg.use_responding_filter else None,
            display_names=display_names,
        )

    # The plots leave out ego (V1): the ship under our control knows its own
    # intent and never acts on the filter's estimate of it.
    if responding_history:
        responding_history = {
            display_names.get(vid, vid): hist for vid, hist in responding_history.items()
            if vid != "ego"
        }

    # Goal markers (viz.goal_markers): each vessel's goal is the straight
    # projection of its initial state along its nominal heading over the full
    # scenario duration — i.e. where it would end up with no interaction.
    goals: Optional[dict[str, tuple[float, float]]] = None
    if vc.goal_markers:
        goals = {
            vi.vessel_id: (
                vi.state.north + vi.state.speed * config.t_horizon * math.cos(vi.state.heading),
                vi.state.east  + vi.state.speed * config.t_horizon * math.sin(vi.state.heading),
            )
            for vi in config.vessels
        }

    sim_info = SimInfo(
        name    = config.name,
        n_steps = config.n_steps,
        dt      = config.dt,
        vessels = [
            VesselMeta(
                vessel_id     = vi.vessel_id,
                initial_state = VesselState(
                    north=vi.state.north, east=vi.state.east,
                    psi=vi.state.heading, speed=vi.state.speed,
                ),
                display_id    = display_names[vi.vessel_id],
            )
            for vi in config.vessels
        ],
    )
    run_live(sim_info, trajectories, vc=vc,
             policy_debug=policy_debug, debug_vessel_ids=debug_vessel_ids,
             pred_fn=pred_fn, nominal_pred_fn=nominal_pred_fn, pred_every=pred_every,
             pred_top_k=pred_cfg.top_k_hypotheses,
             plan_fn=_plan_overlay_fn(plan_log, config.dt) if plan_log else None,
             cost_fn=_plan_costs_fn(plan_log, config.dt, display_names) if plan_log else None,
             colregs_graphs=colregs_graphs,
             responding_history=responding_history or None,
             stop_time=args.stop_time,
             goals=goals)

    # Standalone p(responding)-vs-time figure (viz.responding_plot), cropped to
    # viz.responding_plot_window if set. Independent of the live panel.
    if vc.responding_plot:
        resp_hist = {
            name: hist for name, hist in (responding_history or {}).items() if hist
        }
        if resp_hist:
            colors_by_name = {m.display_id: VESSEL_COLORS[i % len(VESSEL_COLORS)]
                              for i, m in enumerate(sim_info.vessels)}
            save_responding_plot(
                resp_hist, colors_by_name, config.dt, vc, config.name,
                window=vc.responding_plot_window, t0=0.0,
            )
        else:
            print("  Warning: viz.responding_plot is on but there's no responding "
                  "history — set prediction.use_responding_filter: true.")


if __name__ == "__main__":
    main()
