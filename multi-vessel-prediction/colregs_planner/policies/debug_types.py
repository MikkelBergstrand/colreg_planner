"""colregs_planner/policies/debug_types.py

Lightweight per-step policy debug snapshots consumed by the live visualizer.
Populated by policies during select_action(); collected by the simulation loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass
class EllipseDebugData:
    """Clearance ellipse in CTO action space (u₁ = speed, u₂ = v·ω)."""
    c1: float   # centre: speed  [m/s]
    c2: float   # centre: v·ω   [m/s·rad/s]
    a1: float   # semi-axis along u₁ [m/s]
    a2: float   # semi-axis along u₂ [m/s·rad/s]


@dataclass
class VOConeDebugData:
    """Velocity-obstacle cone in (vₙ, vₑ) velocity space."""
    apex_vn:    float   # apex north-velocity [m/s]
    apex_ve:    float   # apex east-velocity  [m/s]
    axis_n:     float   # cone-axis unit vector, north component
    axis_e:     float   # cone-axis unit vector, east  component
    half_angle: float   # half opening angle [rad]


@dataclass
class StarboardConstraintDebugData:
    """Port-side exclusion half-plane for one obstacle in velocity space.

    The admissible region is the half-plane where the relative velocity
    (v − apex) is clockwise of the bearing (i.e. cross-product ≥ 0 in NED).
    The boundary line passes through apex in the bearing direction; the
    excluded (port) side has normal (−bearing_n, bearing_e) in plot (vₑ, vₙ).

    For HEAD_ON obstacles, apex is laterally shifted from obs_vel by the
    anticipated cooperative lateral manoeuvre (rvo_weight * lateral component
    of (v_own − v_obs)), making the constraint more conservative than plain
    obs_vel anchoring.  For all other roles apex equals obs_vel directly.
    """
    apex_vn:   float   # reference velocity north [m/s] (obs_vel or RVO-shifted)
    apex_ve:   float   # reference velocity east  [m/s] (obs_vel or RVO-shifted)
    bearing_n: float   # unit bearing vector north (own → obstacle, NED)
    bearing_e: float   # unit bearing vector east


@dataclass
class CTODebugFrame:
    """Per-step debug snapshot for CTOPolicy."""
    u_nom:       tuple[float, float]                  # nominal  (u₁, u₂) in action space
    u_projected: tuple[float, float]                  # ideal boundary point (pre-clamp)
    u_selected:  tuple[float, float]                  # final command (after ω_max / speed clamp)
    ellipses:    list[EllipseDebugData] = field(default_factory=list)
    ref_speed:   float = 0.0
    omega_max:   float = 0.0                          # physical turn-rate limit [rad/s]


@dataclass
class VODebugFrame:
    """Per-step debug snapshot for VOPolicyKuwata or VOPolicyAnalytic."""
    v_pref_vn:             float                                  # preferred vel. north [m/s]
    v_pref_ve:             float                                  # preferred vel. east  [m/s]
    v_sel_vn:              float                                  # selected  vel. north [m/s]
    v_sel_ve:              float                                  # selected  vel. east  [m/s]
    cones:                 list[VOConeDebugData]              = field(default_factory=list)
    ref_speed:             float                              = 0.0
    fallback_active:       bool                               = False
    v_pref_proj_vn:        float                             = float("nan")
    v_pref_proj_ve:        float                             = float("nan")
    starboard_constraints: list[StarboardConstraintDebugData] = field(default_factory=list)
    forward_unit_n:        float                             = float("nan")  # NaN = no constraint
    forward_unit_e:        float                             = float("nan")


PolicyDebugFrame = Union[CTODebugFrame, VODebugFrame]
