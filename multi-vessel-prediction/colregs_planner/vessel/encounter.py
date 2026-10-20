"""vessel/encounter.py

Encounter geometry, scenario configuration, and build_scenario().

Each target is placed relative to ego using a per-target encounter type.
Ego is always at the NED origin with a configurable heading (default: north).
Any number of targets is supported.

Angles follow NED convention: 0 = north, positive clockwise.

Author: Johannes Robert Skarø
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

from colregs_planner.vessel.state import InitialState, Intent, VesselInit


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EncounterType(Enum):
    """Encounter geometry relative to ego."""
    HEAD_ON            = auto()   # target dead ahead, reciprocal course
    CROSSING_STARBOARD = auto()   # target on ego's starboard side, crossing ahead
    CROSSING_PORT      = auto()   # target on ego's port side, crossing ahead
    OVERTAKING_AHEAD   = auto()   # ego overtakes target (target ahead, slower)
    OVERTAKING_ASTERN  = auto()   # target overtakes ego (target astern, faster)


_ENCOUNTER_STR: dict[str, EncounterType] = {
    "head_on":            EncounterType.HEAD_ON,
    "crossing_starboard": EncounterType.CROSSING_STARBOARD,
    "crossing_port":      EncounterType.CROSSING_PORT,
    "overtaking_ahead":   EncounterType.OVERTAKING_AHEAD,
    "overtaking_astern":  EncounterType.OVERTAKING_ASTERN,
}

_OVERTAKING_LATERAL = 50.0   # [m] starboard offset for overtaking to avoid collinear tracks


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    """Full scenario specification."""

    name:          str
    vessels:       list[VesselInit]
    dt:            float                    = 1.0
    t_horizon:     float                    = 300.0
    noise_std_pos: float                    = 0.0
    noise_std_vel: float                    = 0.0
    seed:          Optional[int]            = 42
    classifier:    dict[str, float]         = field(default_factory=dict)  # ColregsClassifier overrides
    hazards:       list[list[list[float]]]  = field(default_factory=list)  # land/shoal polygons, [north, east] vertices [m]

    @property
    def n_steps(self) -> int:
        return int(self.t_horizon / self.dt)

    @property
    def n_vessels(self) -> int:
        return len(self.vessels)

    @property
    def ego(self) -> VesselInit:
        return self.vessels[0]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _ned_from_ego_frame(
    forward: float, starboard: float, psi_ego: float,
) -> tuple[float, float]:
    """Convert (forward, starboard) in ego body frame to NED (north, east).

    Ego is at the NED origin with heading psi_ego.
    Starboard is 90° clockwise from the heading direction.
    """
    north = forward * np.cos(psi_ego) - starboard * np.sin(psi_ego)
    east  = forward * np.sin(psi_ego) + starboard * np.cos(psi_ego)
    return float(north), float(east)


def _target_init(
    encounter:        EncounterType,
    range_m:          float,
    ego_speed:        float,
    tgt_speed:        float,
    psi_ego:          float,
    lateral_offset_m: Optional[float] = None,
) -> InitialState:
    """Compute a target's InitialState for the given encounter geometry.

    Ego is assumed to be at the NED origin with heading psi_ego.
    For crossing scenarios, target position is timed so both vessels
    reach the intercept point simultaneously.
    lateral_offset_m shifts the target starboard (positive) or port (negative)
    for HEAD_ON encounters, creating a near-miss geometry (default 0). For
    OVERTAKING encounters it replaces the default _OVERTAKING_LATERAL offset.
    """
    if encounter == EncounterType.HEAD_ON:
        north, east = _ned_from_ego_frame(range_m, lateral_offset_m or 0.0, psi_ego)
        heading = _wrap(psi_ego + np.pi)

    elif encounter == EncounterType.CROSSING_STARBOARD:
        # Target on starboard (right), heading across from starboard to port.
        # Placed ahead by the distance ego travels while target closes range_m.
        fwd = range_m * (ego_speed / tgt_speed) if tgt_speed > 1e-6 else range_m
        north, east = _ned_from_ego_frame(fwd, range_m, psi_ego)
        heading = _wrap(psi_ego - np.pi / 2)

    elif encounter == EncounterType.CROSSING_PORT:
        fwd = range_m * (ego_speed / tgt_speed) if tgt_speed > 1e-6 else range_m
        north, east = _ned_from_ego_frame(fwd, -range_m, psi_ego)
        heading = _wrap(psi_ego + np.pi / 2)

    elif encounter == EncounterType.OVERTAKING_AHEAD:
        # Target ahead of ego, same direction, with a lateral offset to starboard
        # so ego can overtake without a head-on collision track.
        lateral = _OVERTAKING_LATERAL if lateral_offset_m is None else lateral_offset_m
        north, east = _ned_from_ego_frame(range_m, lateral, psi_ego)
        heading = psi_ego

    elif encounter == EncounterType.OVERTAKING_ASTERN:
        # Target astern of ego, same direction.
        lateral = _OVERTAKING_LATERAL if lateral_offset_m is None else lateral_offset_m
        north, east = _ned_from_ego_frame(-range_m, lateral, psi_ego)
        heading = psi_ego

    else:
        raise ValueError(f"Unhandled encounter type: {encounter}")

    return InitialState(north=north, east=east, speed=tgt_speed, heading=heading)


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------

def build_scenario(
    ego_speed:     float,
    ego_heading:   float,           # [rad] NED
    target_cfgs:   dict[str, dict], # {vessel_id: {encounter, range, speed, intent, ...}}
    ego_intent:    Intent        = Intent.RESPONSIVE,
    dt:            float         = 1.0,
    t_horizon:     float         = 300.0,
    noise_std_pos: float         = 0.0,
    noise_std_vel: float         = 0.0,
    seed:          Optional[int] = 42,
) -> ScenarioConfig:
    """Build a ScenarioConfig with ego at the NED origin and targets placed
    by encounter-type geometry.

    Parameters
    ----------
    ego_speed    : initial speed of ego [m/s]
    ego_heading  : initial heading of ego [rad, NED]
    target_cfgs  : per-target config dicts; each must contain at minimum
                   'encounter' (str) and 'range' (float, metres).
                   Optional keys: 'speed', 'intent', 'lateral_offset_m'.
    """
    ego = VesselInit(
        vessel_id      = "ego",
        state          = InitialState(
            north=0.0, east=0.0, speed=ego_speed, heading=ego_heading,
        ),
        intent         = ego_intent,
    )

    vessels       = [ego]
    enc_names:     list[str]                = []

    for vessel_id, tcfg in target_cfgs.items():
        enc_str = tcfg.get("encounter", "head_on")
        enc     = _ENCOUNTER_STR.get(enc_str)
        if enc is None:
            raise ValueError(
                f"Unknown encounter '{enc_str}' for vessel '{vessel_id}'. "
                f"Choose: {', '.join(_ENCOUNTER_STR)}"
            )

        range_m   = float(tcfg.get("range", 400.0))
        tgt_speed = float(tcfg.get("speed",   5.0))
        intent    = (Intent.NON_RESPONSIVE
                     if str(tcfg.get("intent", "responsive")).lower() == "non_responsive"
                     else Intent.RESPONSIVE)

        lateral_offset_m = tcfg.get("lateral_offset_m")
        if lateral_offset_m is not None:
            lateral_offset_m = float(lateral_offset_m)
        state = _target_init(enc, range_m, ego_speed, tgt_speed, ego_heading, lateral_offset_m)
        vessels.append(VesselInit(
            vessel_id      = vessel_id,
            state          = state,
            intent         = intent,
        ))
        enc_names.append(enc_str)

    name = " + ".join(enc_names) if enc_names else "scenario"

    return ScenarioConfig(
        name          = name,
        vessels       = vessels,
        dt            = dt,
        t_horizon     = t_horizon,
        noise_std_pos = noise_std_pos,
        noise_std_vel = noise_std_vel,
        seed          = seed,
    )
