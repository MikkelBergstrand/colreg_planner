"""colregs_planner/scenario_config.py

Scenario YAML loading and VO-policy config extraction, shared between
scripts/run_simulation.py (simulation + live playback).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from colregs_planner.colregs_graph.colregs_classifier import ColregsClassifier
from colregs_planner.vessel.encounter import ScenarioConfig, build_scenario
from colregs_planner.vessel.state import VesselInit, Intent
from colregs_planner.prediction.vo_rollout import VOAgentConfig, _RUNTIME_FIELDS
from colregs_planner.seachart.hazard import HazardMap
from colregs_planner.visualization.live_viz import VizConfig


# ---------------------------------------------------------------------------
# Prediction config
# ---------------------------------------------------------------------------

@dataclass
class PredictionConfig:
    enabled:                    bool  = False
    horizon:                    float = 60.0   # [s]
    update_dt:                  float = 5.0    # recompute prediction every this many seconds [s]
    use_responding_filter:      bool  = False  # 1-step predictive responding filter; enumerates
                                               # 2^M hypotheses from responsiveness posteriors
    policy_type:                str   = VOAgentConfig.policy_type    # "vo_analytic" or "vo_kuwata"
    rollout_model:              str   = VOAgentConfig.rollout_model  # motion model for the full-horizon / viz rollout
    filter_rollout_model:       str | None = None  # model for the responding filter's 1-step
                                                   # scoring rollout; None → same as rollout_model
    show_nominal_comparison:    bool  = False  # overlay the dotted nominal (no-interaction) rollout
    reclassify_in_rollout:      bool  = True   # re-run the COLREGs classifier at every rollout step
                                               # (paper Sec. V-D); False = hold each hypothesis graph fixed (faster)
    top_k_hypotheses:           int   = 1      # how many responsiveness-hypothesis trajectory bundles
                                               # to draw live (1 = top-1 only; >1 overlays the next-most-
                                               # likely bundles faded). Needs use_responding_filter.
    parallel_workers:           int | str = "auto"  # worker processes for the top-K hypothesis rollouts
                                               # (0 = serial); "auto" = min(top_k, 2^N, CPU cores - 1),
                                               # i.e. no workers when top_k_hypotheses is 1
    # --- DCPA evaluation (prediction.dcpa_eval block) -----------------------
    # At each listed sim time, roll out a prediction from the true state and
    # compare the predicted distance at closest point of approach (DCPA) with
    # the ground-truth DCPA, for both the top-1 responsiveness hypothesis and the
    # "assume everyone responsive" baseline.
    dcpa_eval_enabled:          bool  = False
    dcpa_eval_times:            tuple = ()          # [s] sim times to evaluate at
    dcpa_eval_horizon:          Optional[float] = None   # [s] rollout/compare window; None → prediction.horizon
    dcpa_eval_pairs:            Optional[list]  = None    # [[id_a, id_b], ...]; None → ego vs every other vessel
    dcpa_eval_out:              Optional[str]   = None    # optional CSV output path

    # prediction.dcpa_eval YAML key -> field
    _DCPA_KEYS = {
        "enabled":   "dcpa_eval_enabled",
        "times_s":   "dcpa_eval_times",
        "horizon_s": "dcpa_eval_horizon",
        "pairs":     "dcpa_eval_pairs",
        "out":       "dcpa_eval_out",
    }

    @classmethod
    def from_dict(cls, p: dict) -> "PredictionConfig":
        """Build from the YAML prediction block.  Missing keys keep the field
        defaults; unknown keys raise a ValueError."""
        p    = dict(p)
        dcpa = p.pop("dcpa_eval", None) or {}
        plain = {f.name for f in fields(cls)} - set(cls._DCPA_KEYS.values())
        unknown = (set(p) - plain) | {f"dcpa_eval.{k}" for k in set(dcpa) - set(cls._DCPA_KEYS)}
        if unknown:
            raise ValueError(
                f"Unknown prediction parameter(s) {sorted(unknown)}. "
                f"Valid: {sorted(plain)} and dcpa_eval.{{{', '.join(cls._DCPA_KEYS)}}}"
            )
        kw = dict(p)
        kw.update({cls._DCPA_KEYS[k]: v for k, v in dcpa.items()})
        if "dcpa_eval_times" in kw:
            kw["dcpa_eval_times"] = tuple(float(t) for t in kw["dcpa_eval_times"] or ())
        return cls(**kw)


# ---------------------------------------------------------------------------
# VO agent config extraction
# ---------------------------------------------------------------------------

# Keys of a simulation vessel block that describe the scenario or the
# simulated vessel rather than VO parameters.
SIM_VESSEL_KEYS = frozenset({
    "speed", "heading", "intent", "encounter", "range", "lateral_offset_m",
    "sim_model", "policy", "turn_rate", "omega_max", "tcpa_method",
    "replan_s", "search", "actual",
})


def sim_vessel_cfg(vcfg: dict) -> dict:
    """The vessel block the simulated vessel itself runs on: vcfg with its
    ``actual`` overrides applied. The predictor keeps using vcfg, so
    ``actual`` is how a vessel really behaves, unknown to the predictor."""
    return {**{k: v for k, v in vcfg.items() if k != "actual"}, **(vcfg.get("actual") or {})}


def _apply_mismatch(vessel_cfgs: dict[str, dict], mismatch: dict) -> None:
    """Fill each target's ``actual`` block from scenario.mismatch.

    mismatch.jitter maps a VO parameter to a relative spread r: the target
    really uses its configured value (or the default) times a factor drawn
    uniformly from [1 - r, 1 + r], with mismatch.seed. Values a vessel sets
    in ``actual`` itself are kept. Ego is the ship whose predictions are
    being tested, so it is left alone.
    """
    unknown = set(mismatch) - {"seed", "jitter"}
    if unknown:
        raise ValueError(f"Unknown scenario.mismatch key(s) {sorted(unknown)}. Valid: ['jitter', 'seed']")
    jitter = mismatch.get("jitter") or {}
    _check_vo_keys(jitter, "scenario.mismatch.jitter")
    rng = np.random.default_rng(mismatch.get("seed", 0))
    for vid, vcfg in sorted(vessel_cfgs.items()):
        if vid == "ego":
            continue
        actual = dict(vcfg.get("actual") or {})
        # Sorted, so a seed draws the same values however the YAML orders its keys.
        for key, r in sorted(jitter.items()):
            base = vcfg.get(key, getattr(VOAgentConfig, key))
            if base is None or not isinstance(base, (int, float)):
                raise ValueError(f"scenario.mismatch.jitter: {key} has no numeric value to jitter")
            value = base * rng.uniform(1.0 - r, 1.0 + r)
            if key not in actual:
                actual[key] = int(round(value)) if isinstance(base, int) else round(float(value), 3)
        if actual:
            vcfg["actual"] = actual


def _check_vo_keys(d: dict, where: str) -> None:
    tunable = {f.name for f in fields(VOAgentConfig)} - _RUNTIME_FIELDS
    unknown = set(d) - tunable
    if unknown:
        raise ValueError(f"Unknown VO parameter(s) {sorted(unknown)} in {where}. Valid: {sorted(tunable)}")


def build_vo_agent_config(vcfg: dict, vi: VesselInit, **overrides) -> VOAgentConfig:
    """VO parameters for one simulated vessel, from its YAML block.

    Tuning keys in vcfg override the VOAgentConfig defaults (see
    VOAgentConfig.from_dict); the vessel's initial state sets the nominal
    course, speed and LOS origin.  overrides (e.g. policy_type,
    rollout_model) take precedence over both.
    """
    return VOAgentConfig.from_dict(
        vcfg,
        ignore       = SIM_VESSEL_KEYS,
        **{
            "nominal_psi":  vi.state.heading,
            "ref_speed":    vi.state.speed,
            "origin_north": vi.state.north,
            "origin_east":  vi.state.east,
            **overrides,
        },
    )


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> tuple[ScenarioConfig, dict[str, dict], VizConfig, PredictionConfig]:
    raw  = yaml.safe_load(path.read_text())
    s    = raw.get("scenario",   {})
    v    = raw.get("viz",        {})
    p    = raw.get("prediction", {})


    vessel_cfgs = s.get("vessels", {})
    for vid, vcfg in vessel_cfgs.items():
        _check_vo_keys(vcfg.get("actual") or {}, f"scenario.vessels.{vid}.actual")
    if s.get("mismatch"):
        _apply_mismatch(vessel_cfgs, s["mismatch"])
    ego_cfg     = vessel_cfgs.get("ego", {})
    target_cfgs = {k: cfg for k, cfg in vessel_cfgs.items() if k != "ego"}

    ego_intent = (Intent.NON_RESPONSIVE
                  if str(ego_cfg.get("intent", "responsive")).lower() == "non_responsive"
                  else Intent.RESPONSIVE)

    config = build_scenario(
        ego_speed       = float(ego_cfg.get("speed",       5.0)),
        ego_heading     = float(np.deg2rad(float(ego_cfg.get("heading", 0.0)))),
        target_cfgs     = target_cfgs,
        ego_intent      = ego_intent,
        dt              = float(s.get("dt",            1.0)),
        t_horizon       = float(s.get("t_horizon",     300.0)),
        noise_std_pos   = float(s.get("noise_std_pos", 0.0)),
        noise_std_vel   = float(s.get("noise_std_vel", 0.0)),
        seed            = s.get("seed", 42),
    )

    # COLREGs classifier distances/gains (scenario.classifier). They are in
    # metres, so a scenario scaled away from ship scale needs to scale them too.
    classifier_cfg = dict(s.get("classifier") or {})
    valid = {f.name for f in fields(ColregsClassifier) if f.init}
    unknown = set(classifier_cfg) - valid
    if unknown:
        raise ValueError(
            f"Unknown scenario.classifier key(s) {sorted(unknown)}. Valid: {sorted(valid)}"
        )
    config.classifier = classifier_cfg
    # Static hazards (scenario.hazards): polygons of [north, east] vertices
    # [m], e.g. the banks of a canal. The search planner and every VO policy
    # built with build_vo_agent_config(..., hazards=config.hazard_map) keep
    # clear of them.
    config.hazards = [[[float(n), float(e)] for n, e in poly] for poly in (s.get("hazards") or [])]
    config.hazard_map = HazardMap.from_polygons(config.hazards) if config.hazards else None

    pred_cfg = PredictionConfig.from_dict(p)

    return config, vessel_cfgs, VizConfig.from_dict(v), pred_cfg
