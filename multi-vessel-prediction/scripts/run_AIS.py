"""run_AIS.py

Play back AIS encounter windows with live vessel-prediction overlay.

Optional panels (controlled via the config's viz block):
  debug_viz: true        — VO velocity-space panel per vessel
  colregs_graph_viz: true — COLREGs role graph
  responding_panel: true  — responsiveness posterior (requires use_responding_filter: true)

All parameters are driven by a YAML config file (default: scenarios/real/window_098.yaml).
--config accepts any config in scenarios/real/ (or elsewhere) — one file per AIS
encounter window; see scenarios/README.md. CLI flags override selected config
values for quick experiments.

Usage
-----
    python scripts/run_AIS.py                          # uses default config
    python scripts/run_AIS.py --config scenarios/real/window_054.yaml
    python scripts/run_AIS.py --config scenarios/real/window_079.yaml --metrics
    python scripts/run_AIS.py --window 260             # play specific window_id
    python scripts/run_AIS.py --list                   # list available windows
    python scripts/run_AIS.py --speed 200              # override realtime_factor
    python scripts/run_AIS.py --start 50 --end 250     # play/plot only frames 50..250
    python scripts/run_AIS.py --record                 # save playback to outputs/videos/
    python scripts/run_AIS.py --save-gif               # save playback to outputs/<config_stem>.gif

Video recording (viz.video in config, or --record)
----------------------------------------------------
  viz.video.enabled:        true/false — record live playback to outputs/videos/
  viz.video.fps:            int        — frame rate of the saved video
  viz.video.start_time_s:   s/null     — restrict the RECORDED video to this sim-time
  viz.video.end_time_s:     s/null       window (t = frame * dt, from the start of the
                                          window). null on either side = unrestricted on
                                          that side. Live playback itself always plays the
                                          full data.start_step..end_step range regardless —
                                          this only trims which frames get grabbed into the
                                          video.

  --save-gif (or viz.save_path in config) records the same live playback to a
  gif instead (outputs/<config_stem>.gif, at viz.fps) — it takes the place of the
  mp4 if both are requested, and honours the same start_time_s/end_time_s window.
  Unlike run_simulation.py, predictions are computed during playback, so the
  gif is grabbed from the live loop rather than rendered offline.

  Frames are grabbed only while playback is running (not while paused), and
  the video is finalized once end_time_s is reached, or on quit/window
  close/Ctrl-C, so an interrupted run still saves whatever played up to that
  point.

Controls
--------
    Space   : pause / resume
    Q / Esc : quit
    S       : save current frame (outputs/frames/; format via viz.frame_format —
              "pdf"/"svg"/"png", default "pdf")

Stop-at-time (viz.stop_time_s in config, or --stop-time)
----------------------------------------------------------
  When playback reaches the given simulation time [s] (t = step * dt, from
  the start of the window), it auto-pauses and saves the current frame —
  handy for grabbing a specific paper figure without hunting for the right
  moment to press S by hand.

Trajectory export (export.* in config, or --export-time)
--------------------------------------------------------
  At a single simulation time [s] (snapped to the nearest frame), dump the
  predicted trajectories and the ground-truth future AIS track for every
  active vessel to a self-describing JSON file under export.out_dir
  (default outputs/data/). The file carries metadata (dt, H, horizon_s,
  window_id, ref point, ...). Live-playback mode only.

    export.enabled:   true/false
    export.at_time_s: sim time [s] to capture (null = off; --export-time overrides)
    export.format:    "json"
    export.out_dir:   output dir relative to the project root

Prediction metrics (metrics.* in config, or --metrics / --metrics-viz)
------------------------------------------------------------------------
  By default, no live playback: step the predictor frame-by-frame and, over
  metrics.start_step..metrics.end_step, score each prediction method against
  the ground-truth future AIS track — ADE / FDE per (frame, vessel, horizon),
  aggregated per method. Ablation over:
    - interaction        : final interaction-aware prediction (predictor.predict())
    - nominal            : nominal-only prediction (predictor.predict_nominal())
    - constant_velocity  : hold the current velocity vector
  Writes <stem>.json (metadata + summary + per-sample) and <stem>.csv (flat
  per-sample rows) to metrics.out_dir (default outputs/data/); with
  metrics.by_step, also <stem>_by_step.csv (mean error vs lookahead step).

  metrics.live_viz: true (or --metrics-viz) runs the SAME scoring alongside
  the normal live view instead of headless, so you can watch the overlaid
  predictions while they're scored — e.g. to confirm the nominal predictor
  (which needs a few steps of position history before its nearest-neighbour
  queries kick in) has warmed up before metrics.start_step. Watch for the
  "Nominal warm-up: V.. ready at frame .." console line per vessel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


from src.vessel.state import VesselState
from src.visualization.live_viz import VesselMeta, SimInfo, VizConfig, LiveVisualizer, save_responding_plot
from src.visualization.multi_panel import PanelSnapshot, save_multi_panel_trajectory_plot
from src.prediction.predictor import VesselPredictor
from src.prediction.parallel import resolve_workers
from src.prediction.vo_rollout import VOAgentConfig
from src.ais.windows import traj_limits, window_to_frames

try:
    from src.nominal.library import TrackLibrary
    from src.nominal.retriever import NominalRetriever
    _NOMINAL_AVAILABLE = True
except ImportError:
    _NOMINAL_AVAILABLE = False

try:
    from src.seachart.loader import find_chart, load_chart_background
    from src.visualization.map_background import render_chart_background, set_water_facecolor
    _SEACHART_AVAILABLE = True
except ImportError:
    _SEACHART_AVAILABLE = False

_ROOT        = Path(__file__).resolve().parents[1]
_DEFAULT_CFG = _ROOT / "scenarios" / "real" / "window_098.yaml"
_FGDB_PATH   = find_chart() if _SEACHART_AVAILABLE else None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_vessel_exclusions(w: dict, exclude_mmsi: list) -> dict:
    """Drop noise vessels from a window's vessel set before playback."""
    excluded = {int(m) for m in exclude_mmsi}
    if not excluded:
        return w
    present = excluded & w["vessels"].keys()
    if present:
        print(f"  Excluding MMSI(s): {sorted(present)}")
    missing = excluded - w["vessels"].keys()
    if missing:
        print(f"  Warning: exclude_mmsi not found in window {w['window_id']}: {sorted(missing)}")
    w["vessels"] = {mmsi: df for mmsi, df in w["vessels"].items() if mmsi not in excluded}
    return w


def _build_predictor(
    cfg: dict, dt: float, ref_lat: float, ref_lon: float,
    *, frame_parallel: bool = False,
) -> tuple[VesselPredictor | None, object | None, bool, bool, bool]:
    """Construct the VesselPredictor (+ nominal retriever) from config. Shared
    by the live playback path (`play`) and the static multi-panel path
    (`multi_panel_run`).

    frame_parallel : the caller runs the rollouts of many frames at the same
        time (metrics_run), so parallel_workers: auto may use all CPU cores
        instead of one worker per top-K hypothesis.

    Returns
    -------
    (predictor, nominal_retriever, interaction_enabled, use_responding_filter)
    """
    pred_cfg        = cfg.get("prediction", {})
    nominal_cfg     = pred_cfg.get("nominal",     {})
    interaction_cfg = pred_cfg.get("interaction", {})
    use_responding_filter  = interaction_cfg.get("use_responding_filter", False)

    predictor: VesselPredictor | None = None
    nominal_retriever = None
    interaction_enabled = interaction_cfg.get("enabled", True)

    if not (nominal_cfg.get("enabled", False) or interaction_cfg.get("enabled", True)):
        return (None, None, interaction_enabled, use_responding_filter)

    if nominal_cfg.get("enabled", False):
        if not _NOMINAL_AVAILABLE:
            print("  Warning: nominal layer requested but src.nominal could not be imported.")
        else:
            lib_path = _ROOT / nominal_cfg["library_path"]
            if lib_path.exists():
                print(f"  Loading nominal library: {lib_path.name} ...")
                library = TrackLibrary.load(lib_path)
                nominal_retriever = NominalRetriever(
                    library,
                    **{k: nominal_cfg[k] for k in ("beta", "r", "max_candidates", "heading_cone_deg")
                       if k in nominal_cfg},
                )
                print(f"  Nominal library: {len(library):,} subtrajectories  "
                      f"(l={library.config.l}, H={library.config.H}, dt={library.config.dt:.0f}s)")
            else:
                print(f"  Warning: nominal library_path not found: {lib_path}")

    agent_cfg = _build_agent_config(interaction_cfg.get("agent", {}))
    # Optional: a distinct motion model for the responding filter's internal
    # 1-step scoring rollout.  Unset → same as agent.rollout_model.
    _agent_cfg_dict = interaction_cfg.get("agent", {})
    responding_rollout_model = _agent_cfg_dict.get("filter_rollout_model")
    top_k     = interaction_cfg.get("top_k_hypotheses", 1)
    n_workers = resolve_workers(
        interaction_cfg.get("parallel_workers", "auto"),
        max_parallel=None if frame_parallel else top_k,
    )
    if n_workers > 0:
        print(f"  Parallel rollouts: {n_workers} worker processes")
    predictor = VesselPredictor(
        dt                        = dt,
        H                         = pred_cfg.get("H", 30),
        default_agent_config      = agent_cfg,
        use_responding_filter            = use_responding_filter,
        responding_rollout_model  = responding_rollout_model,
        default_top_k             = top_k,
        reclassify_in_rollout     = bool(interaction_cfg.get("reclassify_in_rollout", True)),
        parallel_workers          = n_workers,
        nominal_retriever         = nominal_retriever,
        interaction_enabled       = interaction_enabled,
    )
    if nominal_retriever is not None:
        predictor.set_nominal_origin(ref_lat, ref_lon)

    return (predictor, nominal_retriever, interaction_enabled, use_responding_filter)


def _build_agent_config(agent: dict) -> VOAgentConfig:
    """VOAgentConfig from the prediction.interaction.agent block; keys not
    set there keep the VOAgentConfig defaults."""
    return VOAgentConfig.from_dict(
        agent, ignore=("filter_rollout_model",),
    )


# ---------------------------------------------------------------------------
# View-limit helpers
# ---------------------------------------------------------------------------

def _view_limits(
    frames: list[dict[str, VesselState]], viz_cfg: dict, start_step: int
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Resolve the main map's (xlim, ylim), in precedence order:

    1. viz.xlim / viz.ylim  — explicit [min, max] in NED metres; exact manual zoom.
    2. viz.zoom_radius_m    — a square window of this half-width, centered on
                               the vessels' positions at start_step (i.e. the
                               encounter, not wherever the AIS track wanders).
    3. viz.padding          — (default) margin added around the full
                               trajectory span across *all* frames in the
                               window. Note this bounding box is normally
                               much larger than `padding` itself, so nudging
                               `padding` barely moves the view — use
                               zoom_radius_m or xlim/ylim for real zoom.
    """
    xlim_cfg, ylim_cfg = viz_cfg.get("xlim"), viz_cfg.get("ylim")
    if xlim_cfg is not None and ylim_cfg is not None:
        return tuple(xlim_cfg), tuple(ylim_cfg)

    zoom_radius = viz_cfg.get("zoom_radius_m")
    if zoom_radius is not None:
        states = list(frames[start_step].values())
        center_e = sum(s.east  for s in states) / len(states)
        center_n = sum(s.north for s in states) / len(states)
        return (
            (center_e - zoom_radius, center_e + zoom_radius),
            (center_n - zoom_radius, center_n + zoom_radius),
        )

    return traj_limits(frames, padding=viz_cfg.get("padding", 300.0))


# ---------------------------------------------------------------------------
# SimInfo builder
# ---------------------------------------------------------------------------

def _build_sim_info(w: dict, frames: list[dict[str, VesselState]], start_step: int = 0) -> SimInfo:
    m           = w["meta"]
    sorted_mmsi = sorted(w["vessels"].keys())
    vessel_metas = [
        VesselMeta(
            vessel_id     = f"V{i+1}",
            initial_state = frames[start_step][str(mmsi)],
        )
        for i, mmsi in enumerate(sorted_mmsi)
        if str(mmsi) in frames[start_step]
    ]
    return SimInfo(
        name=(
            f"AIS win={w['window_id']}  {m['scenario'].upper()}"
            f"  enc_CPA={m['enc_cpa_m']:.0f} m  TCPA={m.get('tcpa_s', float('inf')):.0f} s"
        ),
        n_steps = len(frames) - 1,
        dt      = w["dt"],
        vessels = vessel_metas,
    )


# ---------------------------------------------------------------------------
# Profiling helper
# ---------------------------------------------------------------------------

def _print_profile(t: dict[str, list[float]]) -> None:
    labels = list(t.keys())
    totals = {k: sum(v) for k, v in t.items()}
    grand  = sum(totals.values())
    means  = {k: totals[k] / len(t[k]) * 1000 for k in labels}
    maxs   = {k: max(t[k]) * 1000 for k in labels}
    n_frames = len(next(iter(t.values())))

    col = max(len(k) for k in labels) + 2
    print(f"\n{'─' * (col + 52)}")
    print(f"  Profiling summary  ({n_frames} frames)")
    print(f"{'─' * (col + 52)}")
    print(f"  {'Operation':<{col}}  {'Total ms':>10}  {'Mean ms':>9}  {'Max ms':>8}  {'Share':>6}")
    print(f"  {'─'*col}  {'─'*10}  {'─'*9}  {'─'*8}  {'─'*6}")
    for k in sorted(labels, key=lambda x: -totals[x]):
        share = totals[k] / grand * 100 if grand > 0 else 0
        print(f"  {k:<{col}}  {totals[k]*1000:>10.1f}  {means[k]:>9.2f}  {maxs[k]:>8.2f}  {share:>5.1f}%")
    print(f"  {'─'*col}  {'─'*10}  {'─'*9}  {'─'*8}  {'─'*6}")
    print(f"  {'TOTAL':<{col}}  {grand*1000:>10.1f}  {grand/n_frames*1000:>9.2f}  {'':>8}  {'100.0%':>6}")
    print(f"{'─' * (col + 52)}\n")


# ---------------------------------------------------------------------------
# Trajectory export  (export.* in config, or --export-time)
# ---------------------------------------------------------------------------

_EXPORT_STATE_FIELDS = ("north", "east", "psi", "speed", "r")


def _state_to_dict(s: VesselState) -> dict[str, float]:
    return {f: float(getattr(s, f)) for f in _EXPORT_STATE_FIELDS}


def _resolve_export_step(
    export_cfg: dict, dt: float, n_frames: int, cli_export_time: float | None,
) -> tuple[int, float] | None:
    """Resolve the (frame index, sim time [s]) at which to dump trajectories.

    Precedence: --export-time  >  export.at_time_s.  Returns None when export
    is disabled or no time is given.
    """
    at_time_s = cli_export_time
    if at_time_s is None:
        if not export_cfg.get("enabled", False):
            return None
        at_time_s = export_cfg.get("at_time_s")
    if at_time_s is None:
        return None

    step = int(round(float(at_time_s) / dt))
    step = max(0, min(step, n_frames - 1))
    snapped = step * dt
    if abs(snapped - float(at_time_s)) > dt / 2 + 1e-9:
        print(f"  Export   : t = {at_time_s} s snapped to {snapped:.1f} s (frame {step})")
    return step, snapped


def _export_trajectories(
    *,
    out_dir:      Path,
    fmt:          str,
    w:            dict,
    cfg:          dict,
    dt:           float,
    export_step:  int,
    export_time_s: float,
    active_states: dict[str, VesselState],
    predicted:    dict[str, list[VesselState]],
    ground_truth: dict[str, list[VesselState]],
    vid_to_mmsi:  dict[str, int],
) -> Path:
    """Write predicted and ground-truth-future trajectories for the active
    vessels at a single time step to ``out_dir`` as a self-describing JSON file.

    predicted[vid]    : list[VesselState], index 0 == the state at export_step
                        (t = export_time_s), one entry per predicted step.
    ground_truth[vid] : list[VesselState] of the true AIS track from the frame
                        AFTER export_step onward, truncated to H.
    """
    pred_cfg = cfg.get("prediction", {})
    H        = pred_cfg.get("H", 30)
    m        = w["meta"]

    def _series(states: list[VesselState], step0: int) -> list[dict]:
        return [
            {"step": step0 + i, "t_s": round((step0 + i) * dt, 3), **_state_to_dict(s)}
            for i, s in enumerate(states)
        ]

    vessels_out: dict[str, dict] = {}
    for vid in sorted(active_states):
        vessels_out[vid] = {
            "mmsi":                vid_to_mmsi.get(vid),
            "current_state":       {"step": export_step,
                                    "t_s": round(export_time_s, 3),
                                    **_state_to_dict(active_states[vid])},
            "predicted":           _series(predicted.get(vid, []), export_step),
            "ground_truth_future": _series(ground_truth.get(vid, []), export_step + 1),
        }

    doc = {
        "metadata": {
            "created_utc":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "window_id":     w["window_id"],
            "scenario":      m.get("scenario"),
            "dt":            float(dt),
            "H":             int(H),                     # prediction horizon in steps
            "horizon_s":     float(H * dt),
            "export_step":   int(export_step),
            "export_time_s": round(float(export_time_s), 3),
            "ref_lat":       w.get("ref_lat"),
            "ref_lon":       w.get("ref_lon"),
            "frame":         "NED",
            "state_fields":  list(_EXPORT_STATE_FIELDS),
            "units":         {"north": "m", "east": "m", "psi": "rad", "speed": "m/s",
                              "r": "rad/s", "t_s": "s"},
            "active_vessels": sorted(active_states),
        },
        "vessels": vessels_out,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"win{w['window_id']:03d}_t{int(round(export_time_s))}s.{fmt}"
    path = out_dir / fname
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    n_pred = sum(1 for v in vessels_out.values() if v["predicted"])
    print(f"  Exported trajectories -> {path}  "
          f"({len(vessels_out)} vessels, {n_pred} with predictions)")
    return path


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

def _start_video_recording(viz: LiveVisualizer, w: dict, vc: VizConfig) -> Path | None:
    """Start recording live playback — to vc.save_path as a gif if set (--save-gif),
    otherwise to an mp4 in outputs/videos/ if viz.video.enabled.

    Returns the output path on success, or None if recording is disabled or
    could not be started (e.g. ffmpeg unavailable) — playback continues either way.
    """
    if vc.save_path:
        path = Path(vc.save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            viz.start_recording(str(path), fps=vc.fps)
        except Exception as exc:
            print(f"  Warning: could not start gif recording: {exc}")
            return None
        print(f"  Recording gif -> {path}  (fps={vc.fps})")
        return path
    if not vc.video_enabled:
        return None
    videos_dir = _ROOT / "outputs" / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    scenario  = w["meta"]["scenario"]
    path = videos_dir / f"win{w['window_id']:03d}_{scenario}_{timestamp}.mp4"
    try:
        viz.start_recording(str(path), fps=vc.video_fps)
    except Exception as exc:
        print(f"  Warning: could not start video recording: {exc}")
        return None
    print(f"  Recording video -> {path}  (fps={vc.video_fps})")
    return path


def _resolve_video_step_range(
    vc: VizConfig, dt: float, start_step: int, last_frame: int,
) -> tuple[int, int]:
    """Resolve the frame range to grab into the recorded video from viz.video.start_time_s /
    end_time_s [s] (t = frame * dt, from the start of the window). None on either side means
    unrestricted on that side: record from wherever playback starts / to wherever it ends.
    Live playback itself is never affected — only which frames get grabbed.
    """
    v_start = (start_step if vc.video_start_time_s is None
               else max(start_step, min(round(vc.video_start_time_s / dt), last_frame)))
    v_end   = (last_frame if vc.video_end_time_s   is None
               else max(start_step, min(round(vc.video_end_time_s   / dt), last_frame)))
    if v_end < v_start:
        v_start, v_end = v_end, v_start
    return v_start, v_end


def play(
    w: dict,
    cfg: dict,
    *,
    profile:    bool  = False,
    start_step: int   = 0,
    end_step:   int | None = None,
    stop_time:  float | None = None,
    export_time: float | None = None,
    collect_metrics: bool = False,
) -> None:
    frames = window_to_frames(w)
    if len(frames) < 2:
        print(f"  Window {w['window_id']}: not enough frames, skipping.")
        return

    # Ground truth for metrics always reads the whole window, even if `frames`
    # is about to be truncated below for the live view's plot extent — grab
    # the reference before any truncation.
    frames_full = frames

    # Optionally truncate the window to end earlier. This also shrinks the
    # auto-computed view extent (see _view_limits / traj_limits), so the
    # early trajectories take up more of the plot.
    if end_step is not None:
        end_clamped = max(2, min(int(end_step), len(frames) - 1))
        if end_clamped < len(frames) - 1:
            print(f"  End      : frame {end_clamped} / {len(frames) - 1} "
                  f"(truncated from {len(frames)} frames)")
        frames = frames[: end_clamped + 1]

    start_step = max(0, min(start_step, len(frames) - 2))

    m               = w["meta"]
    dt              = w["dt"]
    ref_lat         = w["ref_lat"]
    ref_lon         = w["ref_lon"]
    pred_cfg        = cfg.get("prediction", {})
    viz_cfg         = cfg.get("viz", {})
    realtime_factor = cfg["viz"]["realtime_factor"]
    H               = pred_cfg.get("H", 30)

    sorted_mmsi = sorted(w["vessels"].keys())
    vid_map     = {str(mmsi): f"V{i+1}" for i, mmsi in enumerate(sorted_mmsi)}
    vid_to_mmsi = {f"V{i+1}": int(mmsi) for i, mmsi in enumerate(sorted_mmsi)}

    # --- Metrics collection (--metrics-viz / metrics.live_viz) --------------
    # Ground truth is scored against `frames_full` (the untruncated window),
    # same as headless metrics_run — only the live view's plot extent is
    # affected by end_step.
    metrics_plan: dict | None = None
    metrics_per_sample: list[dict] = []
    metrics_step_sum: dict[str, np.ndarray] = {}
    metrics_step_cnt: dict[str, np.ndarray] = {}
    if collect_metrics:
        metrics_plan = _resolve_metrics_plan(
            cfg, dt, H, len(frames_full) - 1, start_step, end_step,
        )
        if not metrics_plan["methods"]:
            print("  metrics: no valid methods — live metrics collection disabled.")
            collect_metrics = False
            metrics_plan = None
        else:
            metrics_step_sum = {mth: np.zeros(metrics_plan["max_h_steps"]) for mth in metrics_plan["methods"]}
            metrics_step_cnt = {mth: np.zeros(metrics_plan["max_h_steps"]) for mth in metrics_plan["methods"]}
            print(f"  Metrics  : scoring frames {metrics_plan['score_start']}..{metrics_plan['score_end']} "
                  f"every {metrics_plan['eval_every']}  methods=[{', '.join(metrics_plan['methods'])}]  "
                  f"-> {metrics_plan['met_cfg'].get('out_dir', 'outputs/data')}/  (watch the overlay "
                  f"to confirm predictions look sane before scoring starts at frame "
                  f"{metrics_plan['score_start']})")

    export_cfg  = cfg.get("export", {})
    export_spec = _resolve_export_step(export_cfg, dt, len(frames), export_time)
    export_done = False

    print(f"\n{'=' * 64}")
    print(f"  Window   : {w['window_id']}")
    print(f"  Scenario : {m['scenario'].upper()}")
    print(f"  enc_CPA  : {m['enc_cpa_m']:.0f} m   TCPA : {m.get('tcpa_s', float('inf')):.0f} s")
    print(f"  Frames   : {len(frames)}  ({len(frames) * dt / 60:.1f} min at {dt:.0f} s/step)")
    if start_step > 0:
        print(f"  Start    : frame {start_step} / {len(frames) - 1}  ({start_step * dt / 60:.1f} min in)")
    if stop_time is not None:
        frame_fmt = viz_cfg.get("frame_format", "pdf")
        print(f"  Stop-at  : t = {stop_time:.1f} s (auto-pause + save {frame_fmt})")
    if export_spec is not None:
        _es, _et = export_spec
        print(f"  Export   : trajectories at t = {_et:.1f} s (frame {_es}) "
              f"-> {export_cfg.get('out_dir', 'outputs/data')}/")
    print(f"  Ref pt   : {ref_lat:.5f} N  {ref_lon:.5f} E")
    print(f"  Vessels  :")
    for i, mmsi in enumerate(sorted_mmsi):
        df = w["vessels"][mmsi]
        print(f"    V{i+1}  MMSI {mmsi}  SOG {df['sog'].min():.1f}–{df['sog'].max():.1f} kn")
    print(f"{'=' * 64}")
    _fmt = viz_cfg.get("frame_format", "pdf")
    print(f"  Controls : SPACE = pause/resume   Q = quit   S = save frame ({_fmt})\n")

    # --- Build VesselPredictor -------------------------------------------
    (predictor, nominal_retriever, interaction_enabled,
     use_responding_filter) = _build_predictor(
        cfg, dt, ref_lat, ref_lon,
    )

    # pred_fn / nominal_pred_fn return cached results so predict() is timed in the loop.
    # _pred_cache holds the top-1 bundle (metrics, export); _bundle_cache all top-K
    # (trajectories, weight) bundles, drawn live when top_k_hypotheses > 1.
    top_k = int(pred_cfg.get("interaction", {}).get("top_k_hypotheses", 1))
    _pred_cache:    list[dict[str, list[VesselState]]] = [{}]
    _bundle_cache:  list[list] = [[]]
    _nominal_cache_ais: list[dict[str, list[VesselState]]] = [{}]

    def pred_fn(__: dict[str, VesselState], _: int):
        return _bundle_cache[0] if top_k > 1 else _pred_cache[0]

    def nominal_pred_fn(__: dict[str, VesselState], _: int) -> dict[str, list[VesselState]]:
        return _nominal_cache_ais[0]

    # --- Optional panel config -------------------------------------------
    vc = VizConfig.from_dict(viz_cfg)

    # Pre-init responding_history so the panel layout slot is reserved. Collected
    # whenever either the live panel or the standalone end-of-run plot wants it.
    responding_history: dict[str, list[float]] = {}
    if predictor is not None and use_responding_filter \
            and (vc.responding_panel or vc.responding_plot):
        responding_history = {f"V{i+1}": [] for i in range(len(sorted_mmsi))}

    show_cg = predictor is not None and vc.colregs_graph_viz

    # --- Build visualizer ------------------------------------------------
    xlim, ylim = _view_limits(frames, viz_cfg, start_step)
    sim_info   = _build_sim_info(w, frames, start_step)

    chart_bg = None
    show_seachart = viz_cfg.get("seachart", True)
    if not show_seachart:
        print("  Sea chart disabled (viz.seachart: false) — plain background, as in simulation plots.")
    if show_seachart and _SEACHART_AVAILABLE and _FGDB_PATH is not None:
        print("  Loading sea chart ...")
        try:
            chart_bg = load_chart_background(_FGDB_PATH, ref_lat, ref_lon, xlim, ylim)
            if not chart_bg.is_empty():
                counts = ", ".join(f"{r}={n}" for r, n in chart_bg.layer_counts.items())
                print(f"  Chart loaded: {counts}")
            else:
                print("  Chart: no features in this viewport.")
        except Exception as exc:
            print(f"  Warning: sea chart load failed: {exc}")

    _show_nominal_comparison = (
        predictor is not None
        and nominal_retriever is not None
        and interaction_enabled
    )
    viz = LiveVisualizer(
        sim_info, vc,
        xlim               = xlim,
        ylim               = ylim,
        pred_fn            = pred_fn if predictor is not None else None,
        nominal_pred_fn    = nominal_pred_fn if _show_nominal_comparison else None,
        pred_every         = pred_cfg.get("pred_every", 1),
        pred_top_k         = top_k,
        show_colregs_graph = show_cg,
        responding_history = responding_history or None,
        responding_history_t0 = start_step * dt,
        show_future_traj   = viz_cfg.get("show_future_traj", False),
        is_ais             = True,
    )
    if chart_bg is not None and not chart_bg.is_empty():
        # water_color: null (default) leaves the plain simulation-plot
        # facecolor in place for open water; set a colour (e.g. "#dfedf5")
        # for the nautical-chart blue tint.
        water_color = viz_cfg.get("water_color")
        if water_color:
            set_water_facecolor(viz.ax, water_color)
        render_chart_background(viz.ax, chart_bg)
    viz.show()

    video_start_step, video_end_step = _resolve_video_step_range(vc, dt, start_step, sim_info.n_steps)
    if vc.video_enabled and (vc.video_start_time_s is not None or vc.video_end_time_s is not None):
        print(f"  Video    : recording restricted to t = {video_start_step * dt:.1f}.."
              f"{video_end_step * dt:.1f} s (frames {video_start_step}..{video_end_step})")
    video_path:   Path | None = None
    video_active: bool        = False
    video_done:   bool        = False

    # --- Playback loop ---------------------------------------------------
    if profile and predictor is not None:
        predictor.enable_profiling()

    frame_pause = dt / realtime_factor
    k = start_step
    stop_time_reached = False
    _nom_warm_reported: set[str] = set()
    _t: dict[str, list[float]] = defaultdict(list)

    def _tick(label: str, t0: float) -> float:
        t1 = time.perf_counter()
        _t[label].append(t1 - t0)
        return t1

    try:
        while k <= sim_info.n_steps and not viz._stopped:
            if not viz._paused:
                t0 = time.perf_counter()
                states = {
                    vid_map[mmsi_str]: frames[k][mmsi_str]
                    for mmsi_str in frames[k]
                    if mmsi_str in vid_map
                }

                colregs_graph = None

                if predictor is not None:
                    t0 = _tick("state_build", t0)
                    p_responsive   = predictor.update(states)
                    t0 = _tick("predictor.update", t0)
                    colregs_graph = predictor._current_nom_graph

                    if nominal_retriever is not None:
                        for vid in states:
                            if vid in _nom_warm_reported:
                                continue
                            nom_res = predictor._nominal_cache.get(vid)
                            if nom_res is not None and nom_res.n_hist > 0:
                                _nom_warm_reported.add(vid)
                                print(f"  Nominal warm-up: {vid} (MMSI {vid_to_mmsi.get(vid)}) "
                                      f"ready at frame {k} (t={k * dt:.1f}s, "
                                      f"n_hist={nom_res.n_hist})")

                    _nominal_cache_ais[0] = predictor.predict_nominal()
                    t0 = _tick("predictor.predict_nominal", t0)

                    bundles = predictor.predict()
                    _pred_cache[0]   = bundles[0][0] if bundles else {}
                    _bundle_cache[0] = bundles
                    t0 = _tick("predictor.predict", t0)

                    for vid, p in p_responsive.items():
                        if vid in responding_history:
                            responding_history[vid].append(p)

                future_trajs = {
                    vid_map[mmsi_str]: [
                        frames[j][mmsi_str]
                        for j in range(k + 1, min(len(frames), k + 1 + H))
                        if mmsi_str in frames[j]
                    ]
                    for mmsi_str in vid_map
                    if mmsi_str in frames[k]
                }
                t0 = _tick("future_traj_build", t0)

                if collect_metrics:
                    scored = (
                        metrics_plan["score_start"] <= k <= metrics_plan["score_end"]
                        and (k - metrics_plan["score_start"]) % metrics_plan["eval_every"] == 0
                    )
                    if scored:
                        preds_by_method: dict[str, dict[str, list[VesselState]]] = {}
                        if "nominal" in metrics_plan["methods"]:
                            preds_by_method["nominal"] = _nominal_cache_ais[0]
                        if "interaction" in metrics_plan["methods"]:
                            preds_by_method["interaction"] = _pred_cache[0]
                        if "constant_velocity" in metrics_plan["methods"]:
                            preds_by_method["constant_velocity"] = {
                                vid: _cv_rollout(st, metrics_plan["max_h_steps"], dt)
                                for vid, st in states.items()
                            }
                        _score_frame_metrics(
                            k, dt, states, preds_by_method, frames_full, vid_to_mmsi,
                            metrics_plan, metrics_step_sum, metrics_step_cnt,
                            metrics_per_sample, w["window_id"],
                            burden_ids=predictor.burden_ids if predictor is not None else None,
                        )
                    t0 = _tick("metrics_score", t0)

                if (vc.video_enabled or vc.save_path) and not video_active and not video_done \
                        and k >= video_start_step:
                    video_path   = _start_video_recording(viz, w, vc)
                    video_active = video_path is not None

                viz.step(states, k, debug_frames=None, colregs_graph=colregs_graph,
                         future_trajs=future_trajs)
                t0 = _tick("viz.step", t0)

                if video_active and k >= video_end_step:
                    viz.stop_recording()
                    print(f"  Video saved: {video_path}")
                    video_active = False
                    video_done   = True

                if stop_time is not None and not stop_time_reached and k * dt >= stop_time:
                    stop_time_reached = True
                    viz.pause()
                    viz._save_frame_pdf()
                    print(f"  Reached stop-time t={k * dt:.1f} s (step {k}) — paused, frame saved.")

                if export_spec is not None and not export_done and k == export_spec[0]:
                    export_done = True
                    _export_trajectories(
                        out_dir       = _ROOT / export_cfg.get("out_dir", "outputs/data"),
                        fmt           = export_cfg.get("format", "json"),
                        w             = w,
                        cfg           = cfg,
                        dt            = dt,
                        export_step   = k,
                        export_time_s = export_spec[1],
                        active_states = states,
                        predicted     = _pred_cache[0],
                        ground_truth  = future_trajs,
                        vid_to_mmsi   = vid_to_mmsi,
                    )

                k += 1
                plt.pause(frame_pause)
                _tick("plt.pause", t0)
            else:
                plt.pause(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        if export_spec is not None and not export_done:
            print(f"  Warning: export requested at frame {export_spec[0]} but playback "
                  f"never reached it (start_step={start_step}) — nothing exported.")
        if video_path is not None and not video_done:
            viz.stop_recording()
            print(f"  Video saved: {video_path}")
        if vc.responding_plot:
            if responding_history:
                save_responding_plot(
                    responding_history, viz._vessel_colors, dt, vc, sim_info.name,
                    window=vc.responding_plot_window, t0=start_step * dt,
                )
            else:
                print("  Warning: viz.responding_plot is on but there's no responding history "
                      "to plot — set prediction.interaction.use_responding_filter: true in config.")
        if collect_metrics:
            if metrics_per_sample:
                _write_metrics_outputs(
                    w, metrics_plan, dt, pred_cfg,
                    metrics_per_sample, metrics_step_sum, metrics_step_cnt,
                )
            else:
                print("  metrics: no scored samples — playback may have stopped "
                      f"before reaching metrics.start_step (frame {metrics_plan['score_start']}), "
                      "or no vessel/horizon combination qualified.")
        if not viz._stopped:
            plt.show()

    if profile and _t:
        if predictor is not None:
            for k_inner, v in predictor.timing_report().items():
                _t[k_inner] = v
        _print_profile(_t)


# ---------------------------------------------------------------------------
# Multi-panel static figure  (viz.multi_panel / --multi-panel)
# ---------------------------------------------------------------------------

def multi_panel_run(
    w:          dict,
    cfg:        dict,
    *,
    start_step: int        = 0,
    end_step:   int | None = None,
) -> None:
    """Build a static N-panel trajectory figure — one panel per sim time [s] in
    ``viz.multi_panel.times`` — instead of the live playback. All panels share
    one y-axis (North); each keeps its own x-axis (East). Only one panel
    (``viz.multi_panel.legend_panel``) carries the legend.

    The predictor is stepped forward frame-by-frame exactly as in `play` so its
    internal state (responding filter, hysteresis) is consistent; the heavier
    predict()/predict_nominal() calls run only at the requested snapshot steps.
    """
    import copy

    frames = window_to_frames(w)
    if len(frames) < 2:
        print(f"  Window {w['window_id']}: not enough frames, skipping.")
        return

    if end_step is not None:
        end_clamped = max(2, min(int(end_step), len(frames) - 1))
        if end_clamped < len(frames) - 1:
            print(f"  End      : frame {end_clamped} / {len(frames) - 1} (truncated)")
        frames = frames[: end_clamped + 1]

    start_step = max(0, min(start_step, len(frames) - 2))

    m            = w["meta"]
    dt           = w["dt"]
    ref_lat      = w["ref_lat"]
    ref_lon      = w["ref_lon"]
    pred_cfg     = cfg.get("prediction", {})
    viz_cfg      = cfg.get("viz", {})
    mp_cfg       = viz_cfg.get("multi_panel", {})
    H            = pred_cfg.get("H", 30)

    raw_times = mp_cfg.get("times")
    if raw_times is None:  # legacy: `steps` was frame indices, not seconds
        raw_times = [s * dt for s in (mp_cfg.get("steps") or [])]
    if not raw_times:
        print("  viz.multi_panel.enabled is true but viz.multi_panel.times is empty "
              "— nothing to plot.")
        return

    last_frame = len(frames) - 1
    t_lo, t_hi = start_step * dt, last_frame * dt
    steps: list[int] = []
    for t in raw_times:
        sc = max(start_step, min(round(float(t) / dt), last_frame))
        if abs(sc * dt - float(t)) > dt / 2 + 1e-9:
            print(f"  multi_panel: t = {t} s clamped to {sc * dt:.1f} s "
                  f"(valid range {t_lo:.1f}..{t_hi:.1f} s)")
        steps.append(sc)

    legend_panel = int(mp_cfg.get("legend_panel", 0))
    legend_loc   = str(mp_cfg.get("legend_loc", "panel"))
    _lncol       = mp_cfg.get("legend_ncol")
    legend_ncol  = int(_lncol) if _lncol else None
    panel_width  = float(mp_cfg.get("panel_width", 4.0))
    # On-screen-sized elements (heading triangle, V-label, line widths) are tuned
    # for the full-width live map; shrink them for the narrower panels. Default
    # scales with panel_width; override with viz.multi_panel.marker_scale.
    _ms = mp_cfg.get("marker_scale")
    marker_scale = float(_ms) if _ms is not None else min(1.0, panel_width / 6.5)

    sorted_mmsi = sorted(w["vessels"].keys())
    vid_map     = {str(mmsi): f"V{i+1}" for i, mmsi in enumerate(sorted_mmsi)}

    print(f"\n{'=' * 64}")
    print(f"  Multi-panel figure — window {w['window_id']}  ({m['scenario'].upper()})")
    print(f"  Panels at steps : {steps}")
    print(f"  Panels at times : {[round(s * dt, 1) for s in steps]} s")
    print(f"  Marker scale    : {marker_scale:.2f}")
    print(f"{'=' * 64}")

    (predictor, nominal_retriever, interaction_enabled,
     _ur) = _build_predictor(cfg, dt, ref_lat, ref_lon)

    show_pred    = predictor is not None
    show_nominal = (
        predictor is not None and nominal_retriever is not None and interaction_enabled
    )
    show_future  = viz_cfg.get("show_future_traj", False)

    # Interaction (COLREGs) graph corner inset in each panel. Defaults to the
    # live view's viz.cg_inset; override with viz.multi_panel.cg_inset.
    _cgi = mp_cfg.get("cg_inset")
    cg_inset = (viz_cfg.get("cg_inset", False) if _cgi is None else bool(_cgi))
    cg_inset = cg_inset and predictor is not None
    cg_inset_side  = str(mp_cfg.get("cg_inset_side")  or viz_cfg.get("cg_inset_side",  "bottom-right"))
    cg_inset_size  = float(mp_cfg.get("cg_inset_size") or viz_cfg.get("cg_inset_size", 0.30))
    _cgs = mp_cfg.get("cg_inset_scale")
    cg_inset_scale = float(_cgs) if _cgs is not None else max(0.55, marker_scale)
    if cg_inset:
        print(f"  Interaction graph inset: {cg_inset_side}  (size {cg_inset_size:.2f})")

    # Wide p(r=R)-vs-time panel over/under the trajectory row, with a marker at
    # each panel's capture time. Needs a responding estimator.
    responding_panel = str(mp_cfg.get("responding_panel", "off")).lower()
    _resp_wanted = responding_panel in ("over", "under", "top", "bottom")
    collect_responding = _resp_wanted and predictor is not None and _ur
    if _resp_wanted and not collect_responding:
        print("  multi_panel: responding_panel is on but there's no responding "
              "estimator — set prediction.interaction.use_responding_filter: true. "
              "Skipping the panel.")
        responding_panel = "off"
    responding_history: dict[str, list[float]] = (
        {vid: [] for vid in vid_map.values()} if collect_responding else {}
    )
    _rw = mp_cfg.get("responding_panel_window") or viz_cfg.get("responding_plot_window")
    responding_window = tuple(_rw) if _rw else None
    responding_ratio = float(mp_cfg.get("responding_panel_ratio", 0.25))
    responding_values = bool(mp_cfg.get("responding_panel_values", True))
    panel_gap = float(mp_cfg.get("panel_gap", 0.012))

    vc         = VizConfig.from_dict(viz_cfg)
    xlim, ylim = _view_limits(frames, viz_cfg, start_step)
    sim_info   = _build_sim_info(w, frames, start_step)

    chart_bg = None
    if viz_cfg.get("seachart", True) and _SEACHART_AVAILABLE and _FGDB_PATH is not None:
        print("  Loading sea chart ...")
        try:
            chart_bg = load_chart_background(_FGDB_PATH, ref_lat, ref_lon, xlim, ylim)
        except Exception as exc:
            print(f"  Warning: sea chart load failed: {exc}")

    # --- step forward, capturing snapshots at the requested steps -----------
    steps_set = set(steps)
    max_step  = max(steps)
    # Keep stepping past the last panel so the responding curve spans the whole
    # time axis, not just up to the last capture time.
    loop_end = max_step
    if collect_responding:
        if responding_window is not None:
            loop_end = min(len(frames) - 1, int(round(responding_window[1] / dt)))
        else:
            loop_end = len(frames) - 1
        loop_end = max(loop_end, max_step)
        if loop_end > max_step:
            print(f"  Responding curve stepped to t = {loop_end * dt:.0f} s "
                  f"(frame {loop_end})")
    trails: dict[str, tuple[list[float], list[float]]] = {
        vid: ([], []) for vid in vid_map.values()
    }
    snap_by_step: dict[int, PanelSnapshot] = {}

    for k in range(start_step, loop_end + 1):
        states = {
            vid_map[mmsi_str]: frames[k][mmsi_str]
            for mmsi_str in frames[k]
            if mmsi_str in vid_map
        }
        for vid, st in states.items():
            trails[vid][0].append(st.east)
            trails[vid][1].append(st.north)

        if predictor is not None:
            p_responsive = predictor.update(states)
            if collect_responding:
                for vid, p in p_responsive.items():
                    if vid in responding_history:
                        responding_history[vid].append(p)

        if k not in steps_set:
            continue

        pred_bundle:  dict[str, list[VesselState]] = {}
        nominal_pred: dict[str, list[VesselState]] = {}
        if predictor is not None:
            if show_nominal:
                nominal_pred = predictor.predict_nominal()
            bundles = predictor.predict()
            pred_bundle = bundles[0][0] if bundles else {}

        future_trajs: dict[str, list[VesselState]] = {}
        if show_future:
            future_trajs = {
                vid_map[mmsi_str]: [
                    frames[j][mmsi_str]
                    for j in range(k + 1, min(len(frames), k + 1 + H))
                    if mmsi_str in frames[j]
                ]
                for mmsi_str in vid_map
                if mmsi_str in frames[k]
            }

        colregs_graph = None
        if cg_inset and predictor is not None:
            colregs_graph = copy.deepcopy(predictor._current_nom_graph)

        snap_by_step[k] = PanelSnapshot(
            step   = k,
            t      = k * dt,
            states = copy.deepcopy(states),
            trails = copy.deepcopy(trails),
            pred   = pred_bundle,
            nominal = nominal_pred,
            future = future_trajs,
            colregs_graph = colregs_graph,
        )

    snapshots = [snap_by_step[s] for s in steps]

    save_multi_panel_trajectory_plot(
        snapshots    = snapshots,
        sim_info     = sim_info,
        vc           = vc,
        xlim         = xlim,
        ylim         = ylim,
        chart_bg     = chart_bg,
        water_color  = viz_cfg.get("water_color"),
        legend_panel = legend_panel,
        legend_loc   = legend_loc,
        legend_ncol  = legend_ncol,
        panel_width  = panel_width,
        marker_scale = marker_scale,
        show_pred    = show_pred,
        show_nominal = show_nominal,
        show_future  = show_future,
        cg_inset       = cg_inset,
        cg_inset_side   = cg_inset_side,
        cg_inset_size   = cg_inset_size,
        cg_inset_scale  = cg_inset_scale,
        responding_panel   = responding_panel,
        responding_history = responding_history or None,
        responding_t0      = start_step * dt,
        responding_dt      = dt,
        responding_window  = responding_window,
        responding_ratio   = responding_ratio,
        responding_values  = responding_values,
        panel_gap          = panel_gap,
        frames_dir   = vc.frames_dir,
        frame_format = vc.frame_format,
        frame_dpi    = vc.frame_dpi,
        sim_name     = sim_info.name,
    )


# ---------------------------------------------------------------------------
# Prediction-accuracy metrics  (metrics.* in config, or --metrics)
#
# ADE / FDE of each prediction method vs the ground-truth future AIS track,
# aggregated over a range of frames.  Ablation: the final interaction-aware
# prediction (predictor.predict()) vs the nominal-only prediction
# (predictor.predict_nominal()) vs a constant-velocity baseline.
# ---------------------------------------------------------------------------

_METRIC_METHODS = ("interaction", "nominal", "constant_velocity")
_METRIC_ALIASES = {
    "cv":                "constant_velocity",
    "cvm":               "constant_velocity",
    "const_vel":         "constant_velocity",
    "constant-velocity": "constant_velocity",
    "interaction_aware": "interaction",
    "interaction-aware": "interaction",
    "final":             "interaction",
    "nominal_only":      "nominal",
}


def _cv_rollout(state: VesselState, n_steps: int, dt: float) -> list[VesselState]:
    """Constant-velocity forward integration: hold the current velocity vector.

    Returns ``n_steps + 1`` states, index 0 == the current state — matching the
    predictor's predict()/predict_nominal() convention (t=0 at index 0).
    """
    vn, ve = state.vn, state.ve
    north, east = state.north, state.east
    out = [state]
    for _ in range(n_steps):
        north += vn * dt
        east  += ve * dt
        out.append(VesselState(north=north, east=east, psi=state.psi,
                               speed=state.speed, r=0.0))
    return out


def _step_errors(pred: list[VesselState], gt: list[VesselState],
                 horizon_steps: int) -> list[float]:
    """Per-step Euclidean displacement error [m], h = 1 .. H.

    ``pred[h]`` is h steps ahead of t=0; ``gt[h-1]`` is the true AIS state at
    that same step (gt[0] is the frame AFTER t=0).  H = min(horizon_steps,
    len(pred)-1, len(gt)).
    """
    H = min(horizon_steps, len(pred) - 1, len(gt))
    return [
        math.hypot(pred[h].north - gt[h - 1].north,
                   pred[h].east  - gt[h - 1].east)
        for h in range(1, H + 1)
    ]


def _max_deviation(pred: list[VesselState], nominal: list[VesselState],
                    horizon_steps: int) -> float | None:
    """Max per-step Euclidean deviation [m] of ``pred`` from ``nominal`` over
    the horizon, h = 0 .. H (both trajectories include t=0 at index 0, same
    predict()/predict_nominal() convention as ``_step_errors``).

    Returns ``None`` if there's no usable overlap (e.g. nominal wasn't
    predicted for this vessel/frame).
    """
    H = min(horizon_steps + 1, len(pred), len(nominal))
    if H < 1:
        return None
    return max(
        math.hypot(pred[t].north - nominal[t].north,
                   pred[t].east  - nominal[t].east)
        for t in range(H)
    )


def _agg(vals: list[float]) -> dict | None:
    if not vals:
        return None
    a = np.asarray(vals, dtype=float)
    return {
        "n":      int(a.size),
        "mean":   float(a.mean()),
        "median": float(np.median(a)),
        "std":    float(a.std(ddof=0)),
        "min":    float(a.min()),
        "max":    float(a.max()),
        "p90":    float(np.percentile(a, 90)),
    }


def _resolve_metrics_plan(
    cfg:          dict,
    dt:           float,
    H:            int,
    last_frame:   int,
    start_step:   int,
    end_step:     int | None,
) -> dict:
    """Resolve ``metrics.*`` config into a concrete scoring plan.

    Shared by the headless path (``metrics_run``) and the live-playback path
    (``play(..., collect_metrics=True)``, i.e. ``--metrics-viz``) so both score
    identically — same frame range, methods, and horizons.
    """
    met_cfg = cfg.get("metrics", {})

    score_start = met_cfg.get("start_step")
    score_start = start_step if score_start is None else max(0, int(score_start))
    _cfg_end = met_cfg.get("end_step")
    if _cfg_end is None:
        _cfg_end = end_step
    score_end   = last_frame if _cfg_end is None else min(int(_cfg_end), last_frame)
    score_start = min(score_start, last_frame - 1)
    eval_every  = met_cfg.get("eval_every")
    eval_every  = int(eval_every) if eval_every else int(cfg.get("prediction", {}).get("pred_every", 1))
    eval_every  = max(1, eval_every)
    loop_start  = min(start_step, score_start)

    raw_methods = met_cfg.get("methods") or list(_METRIC_METHODS)
    methods: list[str] = []
    for mth in raw_methods:
        key = _METRIC_ALIASES.get(str(mth).strip().lower(), str(mth).strip().lower())
        if key not in _METRIC_METHODS:
            print(f"  metrics: unknown method '{mth}' — skipping "
                  f"(valid: {', '.join(_METRIC_METHODS)})")
            continue
        if key not in methods:
            methods.append(key)

    raw_h = met_cfg.get("horizons_s")
    if raw_h:
        horizon_specs: list[tuple[float, int]] = []
        for h in raw_h:
            steps = min(H, max(1, int(round(float(h) / dt))))
            horizon_specs.append((round(steps * dt, 3), steps))
    else:
        horizon_specs = [(round(H * dt, 3), H)]
    # de-dup by step count, keep order
    _seen: set[int] = set()
    horizon_specs = [hs for hs in horizon_specs
                     if not (hs[1] in _seen or _seen.add(hs[1]))]
    max_h_steps = max(hs[1] for hs in horizon_specs) if horizon_specs else H

    return {
        "met_cfg":        met_cfg,
        "score_start":    score_start,
        "score_end":      score_end,
        "eval_every":     eval_every,
        "loop_start":     loop_start,
        "methods":        methods,
        "horizon_specs":  horizon_specs,
        "max_h_steps":    max_h_steps,
        "require_full":   bool(met_cfg.get("require_full_horizon", True)),
        "min_speed":      float(met_cfg.get("min_speed", 0.0)),
        "want_by_step":   bool(met_cfg.get("by_step", True)),
        "need_predictor": ("interaction" in methods) or ("nominal" in methods),
    }


def _score_frame_metrics(
    k:               int,
    dt:              float,
    states:          dict[str, VesselState],
    preds_by_method: dict[str, dict[str, list[VesselState]]],
    frames_full:     list[dict[str, VesselState]],
    vid_to_mmsi:     dict[str, int],
    plan:            dict,
    step_sum:        dict[str, np.ndarray],
    step_cnt:        dict[str, np.ndarray],
    per_sample:      list[dict],
    window_id,
    burden_ids:      set[str] | None = None,
) -> None:
    """Score one already-predicted frame (``preds_by_method``) against the
    ground-truth future AIS (read from ``frames_full``, never truncated),
    appending ADE/FDE rows to ``per_sample`` and accumulating the by-step
    error curve in ``step_sum`` / ``step_cnt`` (mutated in place).

    ``pred[h]`` (h steps ahead of t=0) is compared against ``gt[h-1]`` (gt[0]
    is the frame right after t=0) — same convention as trajectory export.

    ``burden_ids`` (``predictor.burden_ids`` for this frame, or ``None`` if no
    predictor/classifier ran) tags every row with ``has_burden_role``: a
    property of the vessel/frame/encounter geometry, not of the method being
    scored, so it's set the same for interaction/nominal/constant_velocity
    rows alike — omitted entirely when ``burden_ids`` is ``None``.
    """
    max_h_steps   = plan["max_h_steps"]
    min_speed     = plan["min_speed"]
    require_full  = plan["require_full"]
    want_by_step  = plan["want_by_step"]
    horizon_specs = plan["horizon_specs"]

    for vid, st in states.items():
        if st.speed < min_speed:
            continue
        mmsi_str = str(vid_to_mmsi[vid])
        gt: list[VesselState] = []
        for j in range(k + 1, min(len(frames_full), k + 1 + max_h_steps)):
            if mmsi_str in frames_full[j]:
                gt.append(frames_full[j][mmsi_str])
            else:
                break
        if not gt:
            continue

        for method, pred_map in preds_by_method.items():
            traj = pred_map.get(vid)
            if not traj or len(traj) < 2:
                continue

            if want_by_step:
                for i, e in enumerate(_step_errors(traj, gt, max_h_steps)):
                    step_sum[method][i] += e
                    step_cnt[method][i] += 1

            for h_s, h_steps in horizon_specs:
                if require_full and len(gt) < h_steps:
                    continue
                errs = _step_errors(traj, gt, h_steps)
                if not errs:
                    continue
                row = {
                    "window_id":     window_id,
                    "frame":         k,
                    "t_s":           round(k * dt, 3),
                    "vessel_id":     vid,
                    "mmsi":          vid_to_mmsi[vid],
                    "method":        method,
                    "horizon_s":     h_s,
                    "horizon_steps": h_steps,
                    "n_eval_steps":  len(errs),
                    "ade_m":         round(float(np.mean(errs)), 4),
                    "fde_m":         round(float(errs[-1]), 4),
                }
                # Encounter-geometry property, not a per-method one — same
                # value regardless of which method this row is scoring.
                if burden_ids is not None:
                    row["has_burden_role"] = vid in burden_ids
                # How far the interaction-aware prediction deviates from the
                # nominal-only one, same vessel/frame/horizon — not a
                # ground-truth comparison. Only meaningful (and only
                # computed) for the interaction method, and only when the
                # nominal prediction was also run this frame.
                if method == "interaction":
                    nom_traj = preds_by_method.get("nominal", {}).get(vid)
                    if nom_traj:
                        dev_m = _max_deviation(traj, nom_traj, h_steps)
                        if dev_m is not None:
                            row["dev_m"] = round(dev_m, 4)
                per_sample.append(row)


def _write_metrics_outputs(
    w:          dict,
    plan:       dict,
    dt:         float,
    pred_cfg:   dict,
    per_sample: list[dict],
    step_sum:   dict[str, np.ndarray],
    step_cnt:   dict[str, np.ndarray],
) -> list[Path] | None:
    """Aggregate ``per_sample`` / the by-step curve and write CSV + JSON to
    ``metrics.out_dir``. Returns the paths written, or None if there was
    nothing to score.
    """
    methods       = plan["methods"]
    horizon_specs = plan["horizon_specs"]
    max_h_steps   = plan["max_h_steps"]
    want_by_step  = plan["want_by_step"]
    met_cfg       = plan["met_cfg"]
    m             = w["meta"]

    if not per_sample:
        print("  metrics: no scored samples (check the frame range / "
              "require_full_horizon / min_speed).")
        return None

    def _summ(rows: list[dict]) -> dict:
        out: dict[str, dict] = {}
        for method in methods:
            per_h: dict[str, dict] = {}
            for h_s, h_steps in horizon_specs:
                sel = [r for r in rows
                       if r["method"] == method and r["horizon_steps"] == h_steps]
                if not sel:
                    continue
                per_h[f"{h_s:g}"] = {
                    "horizon_s":     h_s,
                    "horizon_steps": h_steps,
                    "n_samples":     len(sel),
                    "ade":           _agg([r["ade_m"] for r in sel]),
                    "fde":           _agg([r["fde_m"] for r in sel]),
                }
            if per_h:
                out[method] = per_h
        return out

    summary          = _summ(per_sample)
    summary_by_vessel = {
        vid: _summ([r for r in per_sample if r["vessel_id"] == vid])
        for vid in sorted({r["vessel_id"] for r in per_sample})
    }

    by_step: dict[str, list[dict]] = {}
    if want_by_step:
        for method in methods:
            curve = []
            for i in range(max_h_steps):
                c = step_cnt[method][i]
                if c <= 0:
                    continue
                curve.append({
                    "step":         i + 1,
                    "t_s":          round((i + 1) * dt, 3),
                    "mean_error_m": round(float(step_sum[method][i] / c), 4),
                    "n":            int(c),
                })
            if curve:
                by_step[method] = curve

    out_dir = _ROOT / met_cfg.get("out_dir", "outputs/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = time.strftime("%Y%m%d_%H%M%S")
    stem = met_cfg.get("out_name") or f"metrics_win{w['window_id']:03d}_{ts}"

    doc = {
        "metadata": {
            "created_utc":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "window_id":       w["window_id"],
            "scenario":        m.get("scenario"),
            "dt":              float(dt),
            "H":               int(pred_cfg.get("H", 30)),
            "score_start":     int(plan["score_start"]),
            "score_end":       int(plan["score_end"]),
            "eval_every":      int(plan["eval_every"]),
            "methods":         methods,
            "horizons":        [{"horizon_s": s, "horizon_steps": n} for s, n in horizon_specs],
            "require_full_horizon": plan["require_full"],
            "min_speed":       plan["min_speed"],
            "ref_lat":         w.get("ref_lat"),
            "ref_lon":         w.get("ref_lon"),
            "frame":           "NED",
            "units":           {"ade_m": "m", "fde_m": "m", "dev_m": "m", "t_s": "s"},
            "prediction_config": pred_cfg,
        },
        "summary":           summary,
        "summary_by_vessel": summary_by_vessel,
        "by_step":           by_step,
        "per_sample":        per_sample,
    }
    json_path = out_dir / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump(doc, f, indent=2)

    csv_path = out_dir / f"{stem}.csv"
    cols = ["window_id", "frame", "t_s", "vessel_id", "mmsi", "method",
            "horizon_s", "horizon_steps", "n_eval_steps", "ade_m", "fde_m", "dev_m",
            "has_burden_role"]
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=cols)
        wtr.writeheader()
        wtr.writerows(per_sample)

    paths_written = [json_path, csv_path]
    if by_step:
        bs_path = out_dir / f"{stem}_by_step.csv"
        with open(bs_path, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["method", "step", "t_s", "mean_error_m", "n"])
            for method, curve in by_step.items():
                for row in curve:
                    wtr.writerow([method, row["step"], row["t_s"],
                                  row["mean_error_m"], row["n"]])
        paths_written.append(bs_path)

    print(f"\n  {'method':<20} {'horizon':>9} {'N':>6} {'ADE [m]':>10} {'FDE [m]':>10}")
    print(f"  {'-'*20} {'-'*9} {'-'*6} {'-'*10} {'-'*10}")
    for method in methods:
        for h_key, h_stat in summary.get(method, {}).items():
            print(f"  {method:<20} {h_key + 's':>9} {h_stat['n_samples']:>6} "
                  f"{h_stat['ade']['mean']:>10.2f} {h_stat['fde']['mean']:>10.2f}")
    print()
    for p in paths_written:
        print(f"  Wrote {p}")
    return paths_written


def metrics_run(
    w:          dict,
    cfg:        dict,
    *,
    start_step: int        = 0,
    end_step:   int | None = None,
) -> None:
    """Score the prediction methods against the ground-truth future AIS over a
    range of frames (headless, no live view) and write CSV + JSON summaries to
    ``metrics.out_dir``. For the same run with the live view attached (e.g. to
    verify the nominal predictor has warmed up before scoring starts), use
    ``play(..., collect_metrics=True)`` / ``--metrics-viz`` instead.

    The predictor is stepped forward frame-by-frame from ``start_step`` (so its
    responding filter / hysteresis state is consistent with a live run); the
    heavier predict()/predict_nominal() calls run only at the scored frames.

    The full-horizon rollouts of a frame do not feed back into the predictor,
    so with ``prediction.interaction.parallel_workers`` > 0 (or "auto": all
    CPU cores but one) they are started
    (start_predict) and the loop moves on; all frames are scored at the end,
    in frame order, so the output is the same as a serial run.
    """
    frames = window_to_frames(w)
    if len(frames) < 3:
        print(f"  Window {w['window_id']}: not enough frames, skipping metrics.")
        return

    # NOTE: unlike live playback, metrics never truncate `frames` — the
    # ground-truth future AIS is always read to the end of the window so the
    # last scored frames still have a full prediction horizon available.
    # `end_step` / --end only cap the last frame that is *scored*.
    start_step = max(0, min(start_step, len(frames) - 2))

    m            = w["meta"]
    dt           = w["dt"]
    ref_lat      = w["ref_lat"]
    ref_lon      = w["ref_lon"]
    pred_cfg   = cfg.get("prediction", {})
    H          = pred_cfg.get("H", 30)
    last_frame = len(frames) - 1

    plan = _resolve_metrics_plan(cfg, dt, H, last_frame, start_step, end_step)
    methods = plan["methods"]
    if not methods:
        print("  metrics: no valid methods — nothing to do.")
        return

    sorted_mmsi = sorted(w["vessels"].keys())
    vid_map     = {str(mmsi): f"V{i+1}" for i, mmsi in enumerate(sorted_mmsi)}
    vid_to_mmsi = {f"V{i+1}": int(mmsi) for i, mmsi in enumerate(sorted_mmsi)}

    print(f"\n{'=' * 64}")
    print(f"  Prediction metrics — window {w['window_id']}  ({m['scenario'].upper()})")
    print(f"  Score frames : {plan['score_start']}..{plan['score_end']} every {plan['eval_every']}  "
          f"(warm-up from {plan['loop_start']})")
    print(f"  Methods      : {', '.join(methods)}")
    print(f"  Horizons     : {', '.join(f'{s:g}s/{n}st' for s, n in plan['horizon_specs'])}"
          f"   (dt={dt:g}s, H={H})")
    print(f"  Full-horizon only : {plan['require_full']}")
    print(f"{'=' * 64}")

    predictor = None
    if plan["need_predictor"]:
        (predictor, _nom_retr, interaction_enabled,
         _ur) = _build_predictor(cfg, dt, ref_lat, ref_lon, frame_parallel=True)
        if predictor is None:
            print("  Warning: could not build a predictor — 'interaction'/'nominal' "
                  "methods will be skipped. Check prediction.* in the config.")
        elif "interaction" in methods and not interaction_enabled:
            print("  Note: prediction.interaction.enabled is false — the "
                  "'interaction' method will match 'nominal' (predict() falls "
                  "back to nominal integration).")

    per_sample: list[dict] = []
    step_sum = {mth: np.zeros(plan["max_h_steps"]) for mth in methods}
    step_cnt = {mth: np.zeros(plan["max_h_steps"]) for mth in methods}

    # Scored frames waiting for their interaction rollouts:
    # (k, states, preds_by_method, pending interaction bundles or None, burden_ids)
    started: list[tuple] = []

    for k in range(plan["loop_start"], plan["score_end"] + 1):
        states = {
            vid_map[s]: frames[k][s]
            for s in frames[k]
            if s in vid_map
        }
        if predictor is not None:
            predictor.update(states)

        scored_frame = (k >= plan["score_start"]) and ((k - plan["score_start"]) % plan["eval_every"] == 0)
        if not scored_frame:
            continue

        preds_by_method: dict[str, dict[str, list[VesselState]]] = {}
        pending = None
        if predictor is not None and plan["need_predictor"]:
            nominal_pred = predictor.predict_nominal()   # also primes predict()
            if "nominal" in methods:
                preds_by_method["nominal"] = nominal_pred
            if "interaction" in methods:
                pending = predictor.start_predict()
        if "constant_velocity" in methods:
            preds_by_method["constant_velocity"] = {
                vid: _cv_rollout(st, plan["max_h_steps"], dt) for vid, st in states.items()
            }
        burden_ids = predictor.burden_ids if predictor is not None else None
        started.append((k, states, preds_by_method, pending, burden_ids))

    for k, states, preds_by_method, pending, burden_ids in started:
        if pending is not None:
            bundles = pending.result()
            # Keep the method order of a serial run (nominal, interaction, cv).
            preds_by_method = {
                **{m: p for m, p in preds_by_method.items() if m != "constant_velocity"},
                "interaction": bundles[0][0] if bundles else {},
                **{m: p for m, p in preds_by_method.items() if m == "constant_velocity"},
            }
        _score_frame_metrics(k, dt, states, preds_by_method, frames, vid_to_mmsi,
                              plan, step_sum, step_cnt, per_sample, w["window_id"],
                              burden_ids=burden_ids)

    _write_metrics_outputs(w, plan, dt, pred_cfg, per_sample, step_sum, step_cnt)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play AIS encounter windows with live prediction overlay.",
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CFG,
                        help=f"YAML config file (default: {_DEFAULT_CFG.name})")
    parser.add_argument("--window", type=int, default=None,
                        help="Override window_id from config")
    parser.add_argument("--speed", type=float, default=None,
                        help="Override realtime_factor from config")
    parser.add_argument("--list", action="store_true",
                        help="List available windows and exit")
    parser.add_argument("--start", type=int, default=None,
                        help="Frame index to start playback from (overrides start_step in config)")
    parser.add_argument("--end", type=int, default=None,
                        help="Frame index to end the run at (overrides end_step in config); "
                             "also shrinks the auto plot extent to the truncated span")
    parser.add_argument("--profile", action="store_true",
                        help="Print per-operation timing summary after playback "
                             "(overrides config; also settable as top-level profile: true)")
    parser.add_argument("--exclude-mmsi", type=str, default=None,
                        help="Comma-separated MMSIs to drop from the scenario "
                             "(overrides exclude_mmsi in config)")
    parser.add_argument("--record", action="store_true",
                        help="Record playback to outputs/videos/ "
                             "(overrides viz.video.enabled in config)")
    parser.add_argument("--save-gif", "--save", dest="save_gif", action="store_true",
                        help="Save playback as a gif to outputs/<config_stem>.gif")
    parser.add_argument("--stop-time", type=float, default=None,
                        help="Simulation time [s] at which to auto-pause and save "
                             "the current frame (format via viz.frame_format; overrides "
                             "viz.stop_time_s in config)")
    parser.add_argument("--export-time", type=float, default=None,
                        help="Simulation time [s] at which to dump predicted + "
                             "ground-truth-future trajectories to export.out_dir "
                             "(overrides export.at_time_s and forces export on)")
    parser.add_argument("--multi-panel", action="store_true",
                        help="Build the static N-panel trajectory figure (one panel "
                             "per sim time [s] in viz.multi_panel.times) instead of "
                             "live playback (overrides viz.multi_panel.enabled)")
    parser.add_argument("--metrics", action="store_true",
                        help="Score the prediction methods (interaction / nominal / "
                             "constant velocity) against the ground-truth future AIS "
                             "over metrics.start_step..metrics.end_step and write "
                             "CSV + JSON to metrics.out_dir instead of live playback "
                             "(overrides metrics.enabled)")
    parser.add_argument("--metrics-viz", action="store_true",
                        help="Run --metrics with the live view attached, so you can "
                             "watch the overlaid predictions (incl. nominal warm-up) "
                             "while they're being scored, instead of headless "
                             "(overrides metrics.live_viz; implies --metrics)")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # --profile always wins; otherwise fall back to config's top-level `profile: true`
    # so it doesn't have to be remembered as a CLI flag on every invocation.
    profile = args.profile or bool(cfg.get("profile", False))

    # CLI overrides
    if args.speed is not None:
        cfg["playback"]["realtime_factor"] = args.speed
        cfg.setdefault("viz", {})["realtime_factor"] = args.speed
    if args.record:
        cfg.setdefault("viz", {}).setdefault("video", {})["enabled"] = True
    if args.save_gif:
        cfg.setdefault("viz", {})["save_path"] = str(_ROOT / "outputs" / f"{args.config.stem}.gif")

    start_step = args.start if args.start is not None else cfg["data"].get("start_step", 0)
    end_step   = args.end   if args.end   is not None else cfg["data"].get("end_step")
    stop_time  = (
        args.stop_time if args.stop_time is not None
        else cfg.get("viz", {}).get("stop_time_s")
    )

    data_cfg = cfg["data"]
    if args.exclude_mmsi is not None:
        exclude_mmsi = [s.strip() for s in args.exclude_mmsi.split(",") if s.strip()]
    else:
        exclude_mmsi = data_cfg.get("exclude_mmsi", []) or []
    pkl_path = _ROOT / data_cfg["pkl_path"]
    if not pkl_path.exists():
        print(f"Error: pkl file not found: {pkl_path}")
        sys.exit(1)

    print(f"Loading {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        all_windows: list[dict] = pickle.load(f)
    print(f"Loaded {len(all_windows)} windows.")

    # Optional review filter
    review: dict[str, str] = {}
    review_path_str = data_cfg.get("review_path", "")
    if review_path_str:
        review_path = _ROOT / review_path_str
        if review_path.exists():
            with open(review_path) as f:
                review = json.load(f)

    filter_accepted = cfg.get("playback", {}).get("filter_accepted_only", False)
    if filter_accepted and review:
        windows = [w for w in all_windows if review.get(str(w["window_id"])) == "accepted"]
        print(f"Filtered to {len(windows)} accepted windows.")
    else:
        windows = all_windows

    if not windows:
        print("No windows to play.")
        sys.exit(1)

    # --list mode
    if args.list:
        print(f"\n{'ID':>6}  {'Scenario':<14}  {'enc_CPA':>8}  {'Vessels':>7}  {'Duration':>8}  Review")
        print("-" * 60)
        for w in windows:
            m   = w["meta"]
            dur = len(next(iter(w["vessels"].values()))) * w["dt"] / 60
            rev = review.get(str(w["window_id"]), "—")
            print(f"{w['window_id']:>6}  {m['scenario']:<14}  {m['enc_cpa_m']:>7.0f}m"
                  f"  {m['n_vessels']:>7}  {dur:>6.1f} min  {rev}")
        return

    # Pick window
    window_id = args.window if args.window is not None else data_cfg.get("window_id")
    if window_id is not None:
        matches = [w for w in windows if w["window_id"] == window_id]
        if not matches:
            print(f"Window {window_id} not found.")
            sys.exit(1)
        w = _apply_vessel_exclusions(matches[0], exclude_mmsi)
    else:
        w = _apply_vessel_exclusions(windows[0], exclude_mmsi)

    run_metrics  = args.metrics or args.metrics_viz or cfg.get("metrics", {}).get("enabled", False)
    live_metrics = run_metrics and (args.metrics_viz or cfg.get("metrics", {}).get("live_viz", False))
    run_multi_panel = (
        args.multi_panel
        or cfg.get("viz", {}).get("multi_panel", {}).get("enabled", False)
    )
    if run_metrics and not live_metrics:
        if args.export_time is not None or cfg.get("export", {}).get("enabled", False):
            print("  Note: trajectory export runs only in live-playback mode, "
                  "not --metrics — skipping export.")
        metrics_run(w, cfg, start_step=start_step, end_step=end_step)
    elif live_metrics:
        if run_multi_panel:
            print("  Note: --metrics-viz runs live playback, not --multi-panel "
                  "— ignoring multi-panel.")
        play(w, cfg, profile=profile, start_step=start_step, end_step=end_step,
             stop_time=stop_time, export_time=args.export_time, collect_metrics=True)
    elif run_multi_panel:
        if args.export_time is not None or cfg.get("export", {}).get("enabled", False):
            print("  Note: trajectory export runs only in live-playback mode, "
                  "not --multi-panel — skipping export.")
        multi_panel_run(w, cfg, start_step=start_step, end_step=end_step)
    else:
        play(w, cfg, profile=profile, start_step=start_step, end_step=end_step,
             stop_time=stop_time, export_time=args.export_time)


if __name__ == "__main__":
    main()
